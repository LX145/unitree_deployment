# Go2 深度感知时序分析：实机 vs 仿真

分析目的：定位实机 depth_e2e 策略「latent 对 gap 响应慢 / 无响应」的原因，
并给出训练侧的改造建议。

结论先行：**实机深度图只有 13.5 Hz 且平均 110 ms 旧，而训练假设是每步一张
新图（50 Hz / ~0 ms）。** 根因是 RealSense SDK 滤波链在 Jetson Orin 上耗时
73 ms/帧，占深度流水线总耗时的 98%。

---

## 1. 测量方法

控制器在 `DepthVelocity` 状态下按 policy step（50 Hz）记录一行 CSV，
开关与路径在 `config.yaml`：

```yaml
DepthVelocity:
  timing_log: true
  timing_log_autostart: false
  timing_log_toggle: LT + start.on_pressed
  timing_log_path: ../../../log/depth_velocity_timing.csv
  timing_log_timestamp: true
```

分析与对比：

```bash
python3 tools/analyze_timing_log.py log/*.csv
```

### 指标定义（两个年龄不能混）

```
depth_age_ms     = t_policy_start - depth_rx_time      # 图像在 buffer 里放了多久
depth_latency_ms = depth_rx_time - depth_capture_time  # 取帧 -> 写入 buffer 的处理耗时
image_age        = depth_age_ms + depth_latency_ms     # ★ 真正的图像年龄
```

`depth_capture_time` 是 `try_wait_for_frames()` 返回该帧的时刻，与 policy 侧
共用 `std::chrono::steady_clock` 基准。

流水线耗时拆解：`update_interval = wait + process + idle`
（`depth_wait_ms` / `depth_process_ms` / 余量），
`depth_filter_ms` 是 `process` 中 SDK 滤波链的部分。

---

## 2. 测量结果

三组均为 12~14 s 的连续行走数据。实机为 Jetson Orin + D435i。

| 指标 | x86 仿真 | 实机 · 滤波开 | 实机 · 滤波关 |
|---|---|---|---|
| **depth 更新率** | 50.00 Hz | **13.48 Hz** | **48.55 Hz** [1] |
| process 总耗时 | — | 74.71 ms | **1.19 ms** |
| ⤷ SDK 滤波部分 | — | 73.35 ms (98.2%) | 0.001 ms (0.1%) |
| ⤷ wait 取帧 | — | 0.006 ms | 0.037 ms |
| **图像年龄 mean** | 2.19 ms | **109.92 ms** | **17.09 ms** |
| 图像年龄 p50 | 2.20 ms | 108.09 ms | 17.70 ms |
| 图像年龄 p95 | 2.68 ms | 150.08 ms | 18.00 ms |
| 图像年龄 max | 2.91 ms | 171.60 ms | 20.46 ms |
| **相邻步复用同一帧** | 0.0% | **73.0%** | **2.9%** |
| 每次写入跳过的相机帧 | — | 4.46 | 1.19 |
| policy step cost | 0.44 ms | 3.54 ms | 3.35 ms |
| 本体感知年龄 | 0.12 ms | 0.59 ms | 0.46 ms |

[1] 按写入次数除以时长统计；中位间隔 20.001 ms，即实际 50.09 Hz，
已顶到 `update_hz: 50` 设定上限。滤波开启时中位间隔为 74.3 ms。

对照组同配置重复测量一致（13.65 Hz / 109.9 ms），结果可复现。

### 要点

- **本体感知不是问题**。`rt/lowstate` 稳定 1000 Hz，policy 启动时只旧 0.5 ms。
- **仿真无法复现该问题**。MuJoCo 侧深度由 raycaster 直接渲染，且被 controller
  的 lowcmd 驱动（`udp_sender.begin_step()`），故相位锁定、恒定 2 ms。
  实机是自由运行的 RealSense，两者不可比。
- **滤波关掉后图像年龄降到 17 ms，复用率降到 2.9%**，深度基本每步都能拿到
  新图（`sensor frames/write` 从 4.46 降到 1.19，约等于每张相机帧都用上）。

---

## 3. 根因

滤波链构造于 `deploy/src/sensors/realsense_depth_camera.cpp`，全部是
librealsense 官方类，顺序与参数照抄 InstinctLab 的
`instinct_onboard/scripts/depth_latent_publisher.py`：

```cpp
rs2::disparity_transform  depth_to_disparity(true);
rs2::disparity_transform  disparity_to_depth(false);
rs2::hole_filling_filter  hole_filling(1);            // farthest from around
rs2::spatial_filter       spatial;
spatial.set_option(RS2_OPTION_FILTER_MAGNITUDE, 5.0f);
spatial.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.75f);
spatial.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 1.0f);
spatial.set_option(RS2_OPTION_HOLES_FILL, 4.0f);
```

在 848×480 下这套链耗时 **73 ms/帧**（host CPU，非设备 ASIC），
`update_hz: 50` 的 20 ms 预算完全装不下，于是流水线退化为处理受限的
13.5 Hz。

**关键推论**：图像在帧末才写入 buffer，而帧是在本次迭代开头取出的，
所以写入时这张图已经旧了 74 ms；再叠加 0~74 ms 的 buffer 相位，
policy 读到时平均 110 ms。

参考：InstinctLab 的 onboard 用的是**同一套官方滤波**，但 latent 只在独立
ROS node 里以 **10 Hz** 发布（100 ms 预算），所以扛得住。我们不是在模仿
它的整套做法，只是借用了“官方滤波链”这一件事。

两点不要误读：

- **分辨率保持 848×480，不要改成 480×270**。训练侧的相机内参是按
  `D435I_DEPTH_INTRINSICS = 848×480` 加 `D435I_DEPTH_ROI = (21, 0, 806, 480)`
  定义的，deploy 侧靠 `crop_for_target_intrinsics()` 把原生图裁到同一视场。
  换分辨率会改变视场/内参映射，从而改变策略实际看到的几何关系。
  我们只需要 58×58 的点采样，原生分辨率主要影响滤波链的补洞精度。
- **降低相机帧率不省 CPU**。滤波成本按“处理的帧数”计，而原本 60 fps 里
  也只处理了 13.5 帧。降 fps 只省 USB 带宽。

deploy 侧新增了三个开关便于逐级排查（默认值保持原行为）：

```yaml
filter_chain: true          # 总开关
filter_disparity: true
filter_hole_filling: true
filter_spatial: true
```

---

## 4. 与训练假设的差异

训练侧
`zenbot_lab/.../tasks/manager_based/locomotion/robots/go2/go2_depth.py`：

```python
# The deployed D435i publishes at 60 Hz while the policy consumes the newest
# available frame at 50 Hz. Render one fresh training frame per policy step.
DEPTH_CAMERA_UPDATE_PERIOD = 0.02
```

即**训练时每个 policy step 都渲染一张新图**。观察项参数：

```python
"history_length": 2,
"history_stride": 1,
"num_output_frames": 1,
"delayed_frame_ranges": (0, 1),
```

```python
# Match InstinctLab's asynchronous-camera approximation: render at the
# policy rate, retain a short history, and give each environment either the
# newest frame or the previous frame for the duration of an episode.
```

`delayed_frame_ranges` 的延迟是**按 episode 固定**采样的（0 或 1 个传感器帧），
不是逐步随机。

### 分布对比

| | 深度图年龄范围 | 均值 |
|---|---|---|
| 训练 (update_period=0.02, 每步刷新, delayed 0/1 帧) | 0 ~ 20 ms | ~10 ms |
| **实机实测（滤波开）** | **—** | **110 ms** |
| 实机（滤波关，现状） | — | 17 ms |

实机滤波开启时的 110 ms 远超训练分布上限，且每张图被重复喂 3.7 步，
GRU 看到的时序统计与训练严重不符。

---

## 5. 关掉滤波的副作用（未决）

关闭 `filter_chain` 后，反光地板区域的 stereo 匹配失败像素会直接暴露出来。
`process_depth()` 把无效值映射成 `max_depth`：

```cpp
if (depth < cfg_.invalid_depth_threshold ||
    (depth <= 0.0f && cfg_.replace_invalid_with_max)) {
    depth = cfg_.max_depth;      // 2.0 m
}
```

归一化后为 `+0.5`，在可视化里呈**黑色**。因此黑色既可能是「无效值」，
也可能是「真实 ≥2 m」，从策略输入图上无法区分。

实机观测到的黑斑**位置固定、形状不规则、边缘模糊**，符合 stereo 匹配失败的
特征（真实几何边界应当是清晰的几何形状），推测为地板反光所致。

**为何值得关注**：训练和仿真都是光线求交，地面永远返回有效深度；
`invalid → max_depth` 的映射本身训练与仿真一致，但「地面出现伪远点」
是训练中不存在的输入分布，encoder 可能把黑斑读成「此处无地面」。

**尚未验证**：黑斑是否真的影响策略表现。判定方法是只开 hole filling
（`filter_hole_filling: true`、`filter_spatial: false`）——SDK 滤波作用在
raw 帧上、在 invalid→max 映射之前，若黑斑消失即可确认是无效值；
同时可读出单独 hole filling 的 `depth_filter_ms` 成本。

---

## 6. 训练改造建议：深度刷新改为 10 Hz

### 6.1 方案

训练侧把 `DEPTH_CAMERA_UPDATE_PERIOD` 从 `0.02` 改为 `0.1`
（即每 5 个 policy step 刷新一张深度图），GRU 仍每步推进。
部署侧对应 `depth_camera.update_hz: 10.0` 并重新开启 `filter_chain: true`。

这样实机可以用官方滤波链补洞，同时把感知吞吐需求降到 1/5。

### 6.2 为什么不影响输出频率

```
深度图 (10Hz) ──> DepthEncoder ──> depth_latent ─┐
                                                  ├─> GRU (50Hz) ──> fusion ──> terrain_latent (50Hz) ──> decoder
本体历史 (50Hz) ──> proprio_encoder ─────────────┘
```

GRU 每步都走，`proprio_latent` 每步更新，因此 hidden state 与 height map
仍是 50 Hz 输出。视觉的**新信息**带宽是 10 Hz，输出在时间上仍连续。

### 6.3 「缓存 latent」与「每步重算」数值等价

`rsl_rl/models/depth_student_actor.py` 中 `_depth_memory()` 无门控，
每步都会调用 `_encode_depth()`。而 `DepthEncoder` 只由
`Conv2d + ELU + Linear` 构成，**没有 BatchNorm、没有 Dropout**，是纯确定性函数。

因此「连续 5 步喂同一张图、每步重算 CNN」与「算一次、缓存复用 5 步」
前向输出逐元素相同；反向传播时 autograd 对重复使用同一张量的梯度也是求和，
两者同样等价。

**结论：训练侧只需修改 `DEPTH_CAMERA_UPDATE_PERIOD`，不需要实现缓存逻辑。**
部署侧目前的 `E2EDepthRunner` 同样无需改动。缓存只是省算力的优化。

### 6.4 延迟预算

`update_hz: 10` 下，实机：

```
循环 = wait(~0) + process(74 ms) + sleep(26 ms) = 100 ms  ->  10 Hz
CPU  = 73 ms/帧 x 10 帧/s = 73% 单核
```

训练与实机的年龄分布：

| | 范围 | 均值 |
|---|---|---|
| 训练 (update_period=0.1, 相位 0~80 ms + delayed 0/100 ms) | 0 ~ 180 ms | ~90 ms |
| 实机 (74 ms 处理 + 0~100 ms 相位) | 74 ~ 174 ms | ~124 ms |

重叠良好，实机上限仍在训练上限之内。但需注意 **73% 单核占用会与 50 Hz
控制回路争抢同一个 SoC**（滤波关闭后 step cost p99 从 10.99 ms 降到 7.05 ms）。

---

## 7. 必须一并修改的约束

### 7.1 `delayed_frame_ranges` 的物理含义会变 5 倍

该参数的单位是**传感器帧**。`update_period` 由 0.02 改为 0.1 后，
同一个 `(0, 1)` 的语义从「0~20 ms」变为「0~100 ms」。

`history_length` 的语义同步变化：`2` 在 0.02 下代表 40 ms 历史，
在 0.1 下代表 200 ms。

### 7.2 扩大 delayed 范围会触发校验失败

`InstinctLab/.../envs/mdp/observations/exteroception.py`：

```python
def check_delay_bounds(self) -> None:
    max_delayed_frames = self.delayed_frame_ranges[1]
    frames_needed_if_no_delay = (self.num_output_frames - 1) * self.history_skip_frames + 1
    if (frames_needed_if_no_delay + max_delayed_frames) > self.sensor_history_length:
        raise ValueError(...)
```

当前 `num_output_frames=1`、`history_skip_frames=1`，故 `frames_needed = 1`：

| delayed_frame_ranges | 需要 history_length | 现值 `2` |
|---|---|---|
| `(0, 1)` | 1 + 1 = 2 | 满足 |
| `(0, 2)` | 1 + 2 = 3 | **报错** |

若要放宽到 `(0, 2)`，必须同时把 `history_length` 提到 3。

### 7.3 保持 `filter_chain_temporal` 关闭

训练（仿真渲染理想深度）没有帧间平滑这种延迟形式，而
`delayed_frame_ranges` 建模的是整帧延迟，两者不等价。建议保持关闭。

---

## 8. 部署侧配置

`deploy/robots/go2/config/policy/velocity/depth_e2e/params/deploy.yaml`：

只改这两项，分辨率、帧率、内参全部维持原样：

```yaml
depth_camera:
  update_hz: 10.0        # 与训练 DEPTH_CAMERA_UPDATE_PERIOD 对齐
  filter_chain: true     # 重新开启官方滤波链以补洞
```

其余保持不变（`raw_width/raw_height = 848/480`、`raw_fps: 60`、
`filter_chain_temporal` 默认 `false`）。

---

## 9. 待确认事项

1. 反光地板产生的无效像素（黑斑）是否真的影响策略表现；
   若影响，方案 6.1 的滤波链正好补上，若不影响则可保持滤波关闭以换取 17 ms 延迟。
2. 训练改为 10 Hz 后，实测实机图像年龄是否落在训练分布内（预期 74~174 ms）。
3. 若 73% 单核占用导致 50 Hz 控制回路抖动加剧，需要做 CPU 亲和性隔离
   （滤波关闭时 policy step cost p99 已从 10.99 ms 降至 7.05 ms，
   说明两者确有争抢）。
4. 评估使用 D400 设备端 processing block
   （`rs2_get_recommended_processing_blocks`）替代 host 端软件滤波，
   可能大幅降低 CPU 占用。

# Unitree Deployment 接入 RealSense D435i 深度图需求说明

本文档用于交给 Claude Code 在 `LX145/unitree_deployment` 仓库中实现“深度图跑酷策略部署”支持。当前仓库已经可以完成盲走策略的 Sim2Sim 和 Sim2Real，目标是在尽量少改动现有 FSM / Unitree SDK2 控制链路的前提下，加入 RealSense D435i 深度图作为策略观测。

---

## 1. 背景与目标

当前 `unitree_deployment` 的控制链路已经能跑盲走策略：

```text
Unitree LowState
  → BaseArticulation / robot->data
  → ObservationManager
  → OrtRunner(policy.onnx)
  → ActionManager
  → processed_actions()
  → LowCmd
```

现在希望加入深度相机 D435i，使部署端支持训练好的深度图跑酷策略。

目标不是引入 ROS2，也不是通过 DDS 传递深度图，而是：

```text
RealSense D435i
  → librealsense2 C++ 本地读取 depth
  → depth preprocessing
  → latest_depth_obs 缓存
  → ObservationManager 作为 depth_image observation 读取
  → ONNX policy 推理
  → Unitree SDK2 发 LowCmd
```

---

## 2. 当前硬件与部署条件

当前 D435i 已经插到 Go2W / Orin 上，并通过官方工具验证正常：

```text
设备：Intel RealSense D435i
USB：USB 3.x / 5000M
librealsense：rs-enumerate-devices 可识别
realsense-viewer：可以正常显示深度图
```

部署代码是 C++ 形式，目标是在 Orin 上运行，不依赖 RealSense ROS wrapper。

---

## 3. 频率设计

训练设置中：

```text
policy frequency：50 Hz
depth update frequency：10 Hz
depth image size：58 × 87
depth type：distance_to_image_plane / z-depth
depth range：[0.0, 2.0] m
normalization：[0, 2.0] → [-0.5, 0.5]
```

部署端也应保持：

```text
PolicyThread：50 Hz
RealSenseDepthThread：10 Hz 更新 latest_depth_obs
LowCmd / FSM 主循环：保持当前实现
```

也就是说，policy 50Hz 推理时连续多次复用最近一帧 depth：

```text
t = 0 ms    depth_0 + proprio_0 → action_0
t = 20 ms   depth_0 + proprio_1 → action_1
t = 40 ms   depth_0 + proprio_2 → action_2
t = 60 ms   depth_0 + proprio_3 → action_3
t = 80 ms   depth_0 + proprio_4 → action_4
t = 100 ms  depth_1 + proprio_5 → action_5
```

不要在 `ObservationManager::compute()` 或 observation term 中阻塞等待相机帧。

---

## 4. 第一阶段实现目标

第一阶段优先实现 **单个 `policy.onnx`** 的接入，不强制拆分 depth encoder 和 actor。

推荐第一阶段 ONNX 输入形式为多输入：

```text
obs   : [1, proprio_dim]
depth : [1, 1, 58, 87]
```

如果策略是 GRU / LSTM，则还需要支持：

```text
h_in  : [num_layers, 1, hidden_dim]
h_out : [num_layers, 1, hidden_dim]
```

如果当前 `policy.onnx` 是单输入 flatten obs，也需要支持：

```text
obs : [1, proprio_dim + depth_dim]
```

但是优先推荐多输入，因为结构更清楚，也方便后续将 depth encoder 拆出来单独 10Hz 推理。

---

## 5. 不建议做的事情

不要做以下改动：

```text
1. 不要引入 ROS2 / realsense2_camera
2. 不要通过 Unitree DDS 自定义消息传完整 depth image
3. 不要在 State_RLBase::run() 里直接读取相机
4. 不要让 policy 线程调用 wait_for_frames() 阻塞
5. 不要重写整个 FSM
6. 不要改变已有盲走策略的默认行为
```

深度图应作为可选功能，通过 `deploy.yaml` 开关控制。

---

## 6. 推荐代码改动位置

### 6.1 CMake 增加 RealSense 支持

目标文件：

```text
deploy/robots/go2w/CMakeLists.txt
```

增加：

```cmake
find_package(realsense2 REQUIRED)
```

并将 `${realsense2_LIBRARY}` 链接到目标中。

如果使用 OpenCV 做 resize / 保存 debug 图片，则增加：

```cmake
find_package(OpenCV REQUIRED)
```

并链接：

```cmake
${OpenCV_LIBS}
```

如果想减少依赖，也可以先手写 nearest-neighbor resize，不强制依赖 OpenCV。

---

### 6.2 在 ArticulationData 中增加 depth buffer

目标文件：

```text
deploy/include/isaaclab/assets/articulation/articulation.h
```

在 `ArticulationData` 中增加：

```cpp
std::vector<float> depth_obs;
std::mutex depth_mtx;
bool depth_valid = false;
double depth_timestamp = 0.0;
```

含义：

```text
depth_obs:
  已经完成预处理、归一化、flatten 后的 policy 输入，不是 raw depth。

depth_valid:
  RealSense 是否已经成功产生过至少一帧有效 depth_obs。

depth_timestamp:
  最近一次 depth_obs 更新时间，用于 debug / 超时保护。
```

注意：需要 `#include <mutex>`。

---

### 6.3 新增 depth observation term

目标文件：

```text
deploy/include/isaaclab/envs/mdp/observations/observations.h
```

增加 observation term：

```cpp
REGISTER_OBSERVATION(depth_image)
{
    auto & data = env->robot->data;

    std::lock_guard<std::mutex> lock(data.depth_mtx);

    int h = params["height"].as<int>(58);
    int w = params["width"].as<int>(87);
    int history = params["history"].as<int>(1);

    if (!data.depth_valid || data.depth_obs.empty()) {
        // 默认返回“远处无障碍”
        // 训练归一化为 [0, 2.0] -> [-0.5, 0.5]
        // max_depth=2.0 对应 0.5
        return std::vector<float>(history * h * w, 0.5f);
    }

    return data.depth_obs;
}
```

注意事项：

```text
1. 不要在这里读取 RealSense。
2. 不要在这里做耗时 preprocessing。
3. 这里只从 robot->data.depth_obs 拿最近缓存。
4. 如果 depth invalid，返回全 0.5，不要返回全 0 或全 -0.5。
```

---

### 6.4 新增 RealSenseDepthCamera 类

推荐新增文件：

```text
deploy/include/sensors/realsense_depth_camera.h
deploy/src/sensors/realsense_depth_camera.cpp
```

类职责：

```text
1. 启动 librealsense2 pipeline
2. 按 10Hz 读取 depth frame
3. 将 raw z16 depth 转成 meter
4. invalid depth 处理
5. resize 到训练分辨率 58×87
6. clip 到 [0.0, 2.0]
7. normalize 到 [-0.5, 0.5]
8. 可选维护 depth history
9. 写入 robot->data.depth_obs
```

建议接口：

```cpp
class RealSenseDepthCamera {
public:
    struct Config {
        bool enable = false;

        int raw_width = 480;
        int raw_height = 270;
        int raw_fps = 60;

        int out_width = 87;
        int out_height = 58;

        int history = 1;

        float min_depth = 0.0f;
        float max_depth = 2.0f;

        float output_min = -0.5f;
        float output_max = 0.5f;

        float update_hz = 10.0f;

        bool replace_invalid_with_max = true;
        bool save_debug_image = false;
    };

    RealSenseDepthCamera(
        const Config& cfg,
        std::shared_ptr<isaaclab::Articulation> robot
    );

    void start();
    void stop();

private:
    void loop();
    std::vector<float> processDepth(
        const uint16_t* raw,
        int raw_w,
        int raw_h,
        float depth_scale
    );

private:
    Config cfg_;
    std::shared_ptr<isaaclab::Articulation> robot_;

    std::thread thread_;
    std::atomic<bool> running_{false};

    std::deque<std::vector<float>> history_;
};
```

---

## 7. Depth preprocessing 具体要求

输入：

```text
RealSense raw depth：uint16 z16
depth_scale：由 depth_sensor.get_depth_scale() 获得
```

转换：

```cpp
float d = raw[i] * depth_scale;
```

invalid depth 处理：

```text
RealSense 无效深度通常为 0。
第一阶段建议将 d <= 0 的点替换为 max_depth。
原因：训练中如果没有模拟 invalid pixel，则“看不到”更应接近远处/无障碍。
```

resize：

```text
raw depth resolution：例如 480×270
policy depth resolution：width=87, height=58
```

注意 OpenCV 的 `cv::Size(width, height)` 顺序：

```cpp
cv::resize(src, dst, cv::Size(87, 58), 0, 0, cv::INTER_NEAREST);
```

如果手写 nearest resize：

```cpp
for (int y = 0; y < out_h; ++y) {
    int src_y = y * raw_h / out_h;
    for (int x = 0; x < out_w; ++x) {
        int src_x = x * raw_w / out_w;
        ...
    }
}
```

clip + normalize：

```cpp
d = std::clamp(d, min_depth, max_depth);

float t = (d - min_depth) / (max_depth - min_depth);
float norm = t * (output_max - output_min) + output_min;
```

默认配置下：

```text
0.0 m → -0.5
2.0 m →  0.5
```

---

## 8. Depth history 要求

如果训练时 depth 输入是单帧：

```text
depth shape = [1, 1, 58, 87]
history = 1
```

如果训练时使用多帧 history：

```text
depth shape = [1, T, 58, 87]
history = T
```

部署端应将 history 展平为：

```text
oldest frame → newest frame
```

即：

```cpp
std::vector<float> stacked;
for (auto& frame : history_) {
    stacked.insert(stacked.end(), frame.begin(), frame.end());
}
```

不要反过来，除非训练时就是 newest-first。

---

## 9. State_RLBase 中集成相机

目标文件：

```text
deploy/include/FSM/State_RLBase.h
deploy/robots/go2w/src/State_RLBase.cpp
```

在 `State_RLBase` 中增加成员：

```cpp
std::unique_ptr<RealSenseDepthCamera> depth_camera;
```

构造函数中读取 `deploy.yaml`：

```cpp
auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");
```

如果存在：

```yaml
depth_camera:
  enable: true
```

则创建 `RealSenseDepthCamera`。

在 `enter()` 中：

```cpp
if (depth_camera) {
    depth_camera->start();
}
```

在 `exit()` 中：

```cpp
if (depth_camera) {
    depth_camera->stop();
}
```

注意：退出 RL state 时要关闭相机线程，避免重复进入状态时启动多个 pipeline。

---

## 10. deploy.yaml 增加配置

建议增加：

```yaml
depth_camera:
  enable: true

  raw_width: 480
  raw_height: 270
  raw_fps: 60

  update_hz: 10.0

  width: 87
  height: 58
  history: 1

  min_depth: 0.0
  max_depth: 2.0

  output_min: -0.5
  output_max: 0.5

  replace_invalid_with_max: true
  save_debug_image: false
```

---

## 11. Observation 配置示例：多输入 ONNX

如果 `policy.onnx` 输入是：

```text
obs
depth
```

则 `deploy.yaml` 里 observations 应为分组形式：

```yaml
observations:
  obs:
    use_gym_history: true
    scale_first: false

    base_ang_vel:
      params: {}
      scale: [1.0, 1.0, 1.0]
      clip: null
      history_length: 1

    projected_gravity:
      params: {}
      scale: [1.0, 1.0, 1.0]
      clip: null
      history_length: 1

    velocity_commands:
      params: {}
      scale: [1.0, 1.0, 1.0]
      clip: null
      history_length: 1

    joint_pos_rel:
      params: {}
      scale: null
      clip: null
      history_length: 1

    joint_vel_rel:
      params: {}
      scale: null
      clip: null
      history_length: 1

    last_action:
      params: {}
      scale: null
      clip: null
      history_length: 1

  depth:
    use_gym_history: false
    scale_first: false

    depth_image:
      params:
        height: 58
        width: 87
        history: 1
      scale: null
      clip: null
      history_length: 1
```

要求：

```text
ONNX input name 必须和 observation group name 一致：
  input "obs"   ← observations.obs
  input "depth" ← observations.depth
```

当前 `OrtRunner` 会读取 ONNX 所有 input name，并在 observation map 中查找同名输入。

---

## 12. Observation 配置示例：单输入 flatten ONNX

如果 `policy.onnx` 只有一个输入：

```text
obs
```

并且 depth 也拼进同一个大向量，则可以写成单组：

```yaml
observations:
  use_gym_history: true
  scale_first: false

  base_ang_vel:
    params: {}
    scale: [1.0, 1.0, 1.0]
    clip: null
    history_length: 1

  projected_gravity:
    params: {}
    scale: [1.0, 1.0, 1.0]
    clip: null
    history_length: 1

  velocity_commands:
    params: {}
    scale: [1.0, 1.0, 1.0]
    clip: null
    history_length: 1

  joint_pos_rel:
    params: {}
    scale: null
    clip: null
    history_length: 1

  joint_vel_rel:
    params: {}
    scale: null
    clip: null
    history_length: 1

  last_action:
    params: {}
    scale: null
    clip: null
    history_length: 1

  depth_image:
    params:
      height: 58
      width: 87
      history: 1
    scale: null
    clip: null
    history_length: 1
```

此时 `ObservationManager` 会输出 `"obs"`。

---

## 13. GRU / LSTM 支持要求

如果 `policy.onnx` 是 recurrent 模型，当前 `OrtRunner` 不够，因为它只支持：

```text
obs → action
```

需要新增 `RecurrentOrtRunner`，支持：

```text
obs, depth, h_in → action, h_out
```

要求：

```text
1. runner 内部维护 hidden_state
2. 初始化 hidden_state 为 0
3. 每次推理传入 h_in
4. 取出 h_out 并缓存为下一步 h_in
5. env reset / state enter / emergency stop 时清零 hidden_state
```

建议第一阶段可以先支持非 recurrent 单步策略，或者导出一个包含 hidden 输入/输出的 ONNX 并实现 `RecurrentOrtRunner`。

---

## 14. Debug 功能要求

实现一个 debug 模式，至少能打印：

```text
depth_valid
depth_obs.size()
depth min / max / mean
depth update frequency
policy inference time
camera thread time
```

可选保存：

```text
real_depth_obs.txt
real_depth_obs.png
```

保存的是 **policy 实际输入的 normalized depth**，不是 raw depth。

可视化映射：

```text
[-0.5, 0.5] → [0, 255]
```

---

## 15. 安全要求

深度图接入后，不能破坏原有安全逻辑。

必须保留：

```text
1. FSM 状态切换
2. Passive / FixStand / RLBase 逻辑
3. lowstate timeout 回 Passive
4. bad_orientation 回 Passive
5. action limit / joint limit / PD gain 逻辑
```

新增安全逻辑：

```text
1. depth 相机未初始化时，depth_image 返回全 0.5
2. depth 超过一定时间未更新时，打印 warning
3. policy action 出现 NaN / inf 时，切 Passive 或禁止发布危险动作
4. RealSense 线程异常退出时，不应导致主控制线程崩溃
```

---

## 16. 测试计划

### 16.1 相机线程独立测试

新增一个可选测试程序或 debug flag：

```bash
./go2w_ctrl --test-depth-camera
```

功能：

```text
1. 启动 RealSense
2. 10Hz 读取并预处理 depth
3. 打印 depth_obs size/min/max/mean
4. 保存 real_depth_obs.png 和 real_depth_obs.txt
5. 不连接机器人 lowcmd
```

如果不方便新增 flag，也可以先在进入 RL state 后打印 depth debug 信息。

---

### 16.2 Policy dry-run 测试

在不真正发危险动作前，验证：

```text
1. depth input shape 与 ONNX 一致
2. ONNX input name 与 deploy.yaml observation group name 一致
3. policy 输出 action size 正确
4. action 没有 NaN / inf
5. target joint position 在合理范围内
```

---

### 16.3 实机上架测试

顺序：

```text
1. Go2W 上架，四脚离地
2. 进入 FixStand
3. 进入 RLBase，但 action scale 先降低
4. 检查关节方向、action 范围、depth 更新频率
5. 确认安全后再逐步恢复正常 action scale
```

---

## 17. 推荐实现顺序

请按以下顺序实现，避免一次改太多：

```text
Step 1:
  CMake 加 realsense2，确认项目能编译。

Step 2:
  新增 RealSenseDepthCamera 类，能单独读 depth 并打印 min/max/mean。

Step 3:
  在 ArticulationData 中加入 depth_obs buffer。

Step 4:
  添加 REGISTER_OBSERVATION(depth_image)。

Step 5:
  在 State_RLBase 中通过 deploy.yaml 开关启动/关闭 depth camera。

Step 6:
  添加 deploy.yaml 的 depth_camera 配置和 depth_image observation。

Step 7:
  跑单输入或多输入 policy.onnx。

Step 8:
  如有 GRU，再实现 RecurrentOrtRunner。

Step 9:
  做 debug 保存 real_depth_obs，并与 IsaacLab sim_depth_obs 对比。
```

---

## 18. 最终验收标准

实现完成后应满足：

```text
1. 不接 D435i 时，盲走策略仍然可以正常编译和运行。
2. depth_camera.enable=false 时，不启动 RealSense。
3. depth_camera.enable=true 时，D435i 以 10Hz 更新 depth_obs。
4. policy 50Hz 运行时读取最近一次 depth_obs，不阻塞相机。
5. depth_obs shape、range、history 顺序与训练一致。
6. ONNX 多输入 obs/depth 可以正常推理。
7. debug 输出能显示 depth min/max/mean 和更新时间。
8. 相机掉帧时不会导致控制线程卡死。
9. 实机 lowcmd 发送逻辑保持原有结构。
```

---

## 19. 关键设计总结

本次需求的核心是：

```text
深度图不是新的通信链路，而是新的 observation source。
```

正确架构是：

```text
RealSenseDepthCamera thread
  → robot->data.depth_obs
  → REGISTER_OBSERVATION(depth_image)
  → ObservationManager
  → OrtRunner / RecurrentOrtRunner
  → ActionManager
  → LowCmd
```

不要把深度图读取逻辑写进 `State_RLBase::run()`，也不要让 observation term 阻塞等待相机。

---

## 20. 补充测试需求：盲走策略运行时测试深度相机

除独立的 `--test-depth-camera` 之外，还需要支持在**部署盲走策略时同时测试深度相机**。目的是在不更换 policy、不改变原有盲走控制逻辑的情况下，验证 RealSense D435i 在实机控制程序中的读取、预处理、刷新频率和线程稳定性。

这个测试非常重要，因为相机独立测试能正常工作，不代表它和 Unitree SDK2、FSM、policy thread 同时运行时也稳定。

---

### 20.1 Blind Policy + Depth Monitor 模式

增加一个可选配置或命令行参数：

```bash
./go2w_ctrl --network eth0 --depth-monitor
```

或者在 `deploy.yaml` 中增加：

```yaml
depth_camera:
  enable: true
  monitor_only: true
```

含义：

```text
enable: true
  启动 RealSenseDepthCamera 线程。

monitor_only: true
  只读取和预处理 depth，不把 depth_image 加入当前盲走 policy 输入。
  盲走 policy 仍然按照原有 observation 和 policy.onnx 运行。
```

也就是说，该模式下控制链路为：

```text
LowState
  → 原有 blind observation
  → blind policy.onnx
  → LowCmd

RealSenseDepthCamera thread
  → depth preprocessing
  → debug statistics / optional save
```

要求：

```text
1. 不改变原有盲走策略输入维度。
2. 不要求盲走 policy.onnx 有 depth 输入。
3. 不影响原有 blind policy 的推理频率。
4. 不影响 Unitree lowcmd 发布。
5. 相机失败时只 warning，不应导致盲走控制程序崩溃。
```

---

### 20.2 深度图刷新频率测试

RealSenseDepthCamera 线程需要统计以下频率：

```text
raw_frame_hz:
  RealSense pipeline 实际拿到 raw depth frame 的频率。

processed_depth_hz:
  实际完成 preprocessing 并更新 robot->data.depth_obs 的频率。

policy_read_depth_hz:
  policy / observation 侧读取 depth_obs 的频率，如果当前 policy 不使用 depth，则该项可为 0。
```

对于训练设定：

```text
目标 processed_depth_hz = 10 Hz
允许范围：9.0 ~ 11.0 Hz
```

建议每 1 秒打印一次：

```text
[DepthMonitor] raw=59.8Hz processed=10.0Hz age=18.4ms valid=1 size=5046 min=-0.50 max=0.50 mean=0.23
```

字段含义：

```text
raw:
  RealSense 实际取帧频率。

processed:
  depth_obs 实际刷新频率。

age:
  当前时刻距离最近一次 depth_obs 更新的时间，单位 ms。

valid:
  depth_obs 是否有效。

size:
  depth_obs vector 长度。单帧 58×87 应为 5046。

min / max / mean:
  normalized depth_obs 的统计量。
```

如果 `processed_depth_hz` 长时间低于目标频率，例如低于 8Hz，需要打印 warning：

```text
[DepthMonitor][WARN] processed depth frequency too low: 7.4 Hz, target 10 Hz
```

如果 `age` 超过 300ms，需要打印 warning：

```text
[DepthMonitor][WARN] depth frame stale: age=342ms
```

---

### 20.3 控制线程干扰测试

在 blind policy 正常运行时，增加周期性统计：

```text
policy_loop_hz
policy_inference_time_ms
fsm_loop_hz 或 lowcmd_publish_hz
depth_thread_hz
```

目的是确认启动 RealSense 后不会明显影响盲走部署。

建议日志示例：

```text
[Runtime] policy=50.0Hz infer=0.42ms lowcmd=1000Hz depth_processed=10.0Hz
```

验收标准：

```text
1. 开启 depth monitor 后，blind policy 仍然稳定在原有 step_dt 对应频率。
2. policy inference time 不应出现明显抖动。
3. lowcmd 发布不应被 RealSense 线程阻塞。
4. depth preprocessing 不应发生在 FSM 1kHz run() 内。
```

---

### 20.4 Depth Observation Size / Shape 测试

增加检查函数，启动时验证：

```text
depth_obs.size() == history * height * width
```

例如：

```text
history=1, height=58, width=87
expected size = 5046
```

如果不一致，打印错误：

```text
[DepthMonitor][ERROR] depth_obs size mismatch: got 4800, expected 5046
```

在 monitor_only 模式下，该错误不应影响盲走策略运行，但需要清晰提示。

在 depth policy 模式下，该错误应阻止进入 RLBase 或直接切回 Passive，避免 ONNX 输入维度错误导致未定义行为。

---

### 20.5 Depth Range 测试

每次打印统计时检查：

```text
normalized depth min >= -0.5 - eps
normalized depth max <=  0.5 + eps
```

如果超过范围，说明 preprocessing 中 clip / normalize 出错。

warning 示例：

```text
[DepthMonitor][WARN] normalized depth out of range: min=-1.23 max=0.50
```

同时检查 invalid depth 比例：

```text
invalid_raw_ratio:
  raw depth 中 d <= 0 的像素比例。
```

建议输出：

```text
[DepthMonitor] invalid_raw=2.3%
```

如果 invalid 比例过高，例如超过 30%，打印 warning：

```text
[DepthMonitor][WARN] too many invalid depth pixels: 42.5%
```

---

### 20.6 保存调试帧

增加可选配置：

```yaml
depth_camera:
  save_debug_image: true
  debug_save_interval_s: 2.0
  debug_save_dir: "/tmp/go2_depth_debug"
```

功能：

```text
每隔 debug_save_interval_s 保存一次：
  raw_depth_u16.png
  depth_m_vis.png
  depth_obs_norm.png
  depth_obs.txt 或 depth_obs.bin
```

建议文件名带时间戳或序号：

```text
/tmp/go2_depth_debug/depth_obs_000001.png
/tmp/go2_depth_debug/depth_obs_000001.txt
```

注意：

```text
1. 默认 save_debug_image=false。
2. 保存图片不能阻塞控制线程。
3. 如果保存频率较低，例如 0.5Hz，可以直接在 depth thread 中做。
4. 不要在 1kHz FSM run() 中写文件。
```

---

### 20.7 盲走时相机开关测试

需要支持以下几种运行组合：

```text
Case A:
  depth_camera.enable=false
  原有盲走策略正常运行。

Case B:
  depth_camera.enable=true
  depth_camera.monitor_only=true
  原有盲走策略正常运行，同时打印 depth monitor。

Case C:
  depth_camera.enable=true
  depth_camera.monitor_only=false
  depth policy 使用 depth_image observation。
```

验收要求：

```text
Case A 不应受到任何影响。
Case B 用于实机安全测试相机线程。
Case C 用于真正视觉策略部署。
```

---

### 20.8 相机掉线 / 拔插测试

在 blind policy + depth monitor 模式下测试：

```text
1. 启动程序并进入盲走 policy。
2. 拔掉 D435i。
3. 程序应打印 RealSense warning。
4. 盲走控制应继续运行或安全切 Passive，取决于配置。
5. 不允许进程直接崩溃。
```

推荐配置项：

```yaml
depth_camera:
  fail_behavior: "warn_only"   # monitor_only 模式默认
```

未来 depth policy 模式下可支持：

```yaml
depth_camera:
  fail_behavior: "passive"     # depth policy 模式推荐
```

行为定义：

```text
warn_only:
  相机失败只打印 warning，depth_valid=false，返回默认全 0.5。

passive:
  相机失败或 depth stale 超过阈值时，FSM 切回 Passive。
```

---

### 20.9 与训练配置一致性检查

程序启动时打印当前 depth 配置：

```text
[DepthConfig]
  raw_width=480 raw_height=270 raw_fps=60
  update_hz=10
  out_width=87 out_height=58 history=1
  min_depth=0.0 max_depth=2.0
  output_min=-0.5 output_max=0.5
  replace_invalid_with_max=true
```

这样可以在实机日志中直接确认部署配置是否与训练一致。

如果未来从 `deploy.yaml` 中读取这些值，需要在启动时完整打印。

---

### 20.10 推荐新增命令行参数

如果当前参数系统方便扩展，建议新增：

```bash
--depth-monitor
--test-depth-camera
--save-depth-debug
```

含义：

```text
--depth-monitor:
  在正常控制程序中启动 RealSenseDepthCamera，但不要求 policy 使用 depth。

--test-depth-camera:
  只启动 RealSenseDepthCamera，不进入 FSM，不发布 LowCmd。

--save-depth-debug:
  保存 debug depth image / tensor。
```

如果不方便新增命令行参数，也可以全部通过 `deploy.yaml` 控制。

---

### 20.11 Blind 策略部署阶段的推荐测试流程

在接入 depth policy 之前，必须完成以下测试：

```text
1. 运行原始 blind policy，确认未改坏。
2. 运行 blind policy + depth_monitor，机器人上架，检查：
   - blind policy 仍然正常
   - depth raw frequency 正常
   - processed depth frequency 约 10Hz
   - depth_obs size = 5046
   - min/max/mean 合理
   - age 不持续超过 300ms
3. 手动遮挡 / 移动物体，检查 depth mean 和 debug image 有明显变化。
4. 保存 depth_obs.png，scp 回主机确认图像方向、上下左右、地面位置正确。
5. 连续运行 5~10 分钟，确认没有内存增长、线程退出或频率下降。
6. 再切换到真正 depth policy。
```

这一阶段的目标是验证：

```text
RealSense 与现有 Unitree SDK2 控制程序可以稳定共存。
```

而不是验证视觉策略效果。


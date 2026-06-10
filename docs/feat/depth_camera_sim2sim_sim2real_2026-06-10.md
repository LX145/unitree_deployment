# 深度相机 Sim2Sim/Sim2Real 无缝接入 — 完整方案

**日期**: 2026-06-10

## 概述

在现有盲走策略部署框架中，新增深度相机（Intel RealSense D435i）的完整支持，包括：
- **Sim2Real**：真机 Orin 上通过 USB 直连 RealSense，采集深度图写入共享 buffer
- **Sim2Sim**：MuJoCo 仿真中通过 `mj_ray()` CPU 射线投射模拟深度相机，经 DDS `rt/depth_image` 发布
- **策略接入**：通过 `REGISTER_OBSERVATION(depth_image)` + 多组观测，将深度图作为独立 ONNX 输入

核心设计原则：**深度图和 IMU、关节数据一样，是"本体感知"的一等公民**。策略代码不感知数据来源（仿真还是真机），同一份 deploy 二进制通过编译期宏 `HAS_REALSENSE` 自动切换数据源。

---

## 架构全貌

```
┌──────────────────────────────────────────────────────────────────────┐
│                      deploy 代码 (go2_ctrl)                           │
│                                                                      │
│  DepthProvider (抽象接口)                                             │
│    ├─ Sim2Real: RealSenseDepthCamera  ──USB──→ D435i                 │
│    └─ Sim2Sim:  DDSDepthProvider       ──DDS──→ rt/depth_image       │
│                         │                                            │
│                         ▼                                            │
│              robot->data.depth_obs  (mutex 保护)                      │
│                         │                                            │
│                         ▼                                            │
│         REGISTER_OBSERVATION(depth_image)                            │
│                         │                                            │
│                         ▼                                            │
│              ObservationManager::compute()                           │
│              → obs_map["depth"] = 5046 维向量                         │
│                         │                                            │
│                         ▼                                            │
│              OrtRunner::act(obs_map) → ONNX 多输入推理                │
└──────────────────────────────────────────────────────────────────────┘

Sim2Sim 数据流:
  MuJoCo mj_ray() → clip/normalize → DDS rt/depth_image
       ↑ 87×58 @ 10Hz                        ↓
  DepthCameraPublisher              DDSDepthProvider → depth_obs

Sim2Real 数据流:
  RealSense D435i → USB 60Hz → process_depth() → depth_obs
       ↑ 480×270 raw              ↓ 87×58 @ 10Hz
  RealSenseDepthCamera::capture_loop()
```

---

## 新增文件

### Deploy 侧 (unitree_deploy)

| 文件 | 说明 |
|------|------|
| `deploy/include/sensors/depth_provider.h` | `DepthProvider` 抽象基类（virtual start/stop/is_running） |
| `deploy/include/sensors/dds_depth_provider.h` | DDS 深度订阅器，订阅 `rt/depth_image` 写入 `depth_obs` |
| `deploy/robots/go2/test_depth_dds.cpp` | DDS 深度校验工具：订阅 + 实时 OpenCV 显示 + PGM 保存 |
| `deploy/robots/go2/config/policy/velocity/cts_amp/params/deploy.yaml` | 多组观测示例配置（obs + depth 双输入） |

### MuJoCo 侧 (unitree_mujoco)

| 文件 | 说明 |
|------|------|
| `simulate/src/depth_camera_publisher.h` | `DepthCameraPublisher` 类：`mj_ray()` CPU 射线投射 + 预处理 + DDS 发布 |
| `simulate/src/param.h` | 新增 `DepthCameraConfig` 结构体（pinhole 模型参数） |
| `simulate/config.yaml` | 新增 `depth_camera` 配置段 |

---

## 修改文件

### Deploy 侧

| 文件 | 改动 |
|------|------|
| `deploy/include/sensors/realsense_depth_camera.h` | 继承 `DepthProvider` |
| `deploy/include/isaaclab/assets/articulation/articulation.h` | 已有 `depth_obs`/`depth_mtx`/`depth_valid`（Phase 1），无改动 |
| `deploy/include/FSM/State_RLBase.h` | `shared_ptr<void>` → `shared_ptr<DepthProvider>` |
| `deploy/robots/go2/src/State_RLBase.cpp` | `#ifdef HAS_REALSENSE` 自动选择 RealSense 或 DDS provider |
| `deploy/robots/go2w/src/State_RLBase.cpp` | 同上 |
| `deploy/robots/g1_29dof/src/State_RLBase.cpp` | 同上 |
| `deploy/include/isaaclab/envs/mdp/observations/observations.h` | 新增 `REGISTER_OBSERVATION(depth_image)` |
| `deploy/robots/go2/CMakeLists.txt` | 新增 `test_depth_dds` 编译目标 |

### MuJoCo 侧

| 文件 | 改动 |
|------|------|
| `simulate/src/unitree_sdk2_bridge.h` | `RobotBridge` 集成 `DepthCameraPublisher` |
| `simulate/src/main.cc` | OpenCV 深度图显示线程、全局指针、glfw hack |
| `simulate/CMakeLists.txt` | 新增 OpenCV 检测 + `HAS_OPENCV` 编译定义 |
| `unitree_robots/go2/go2.xml` | 新增 `<site name="depth_camera">` 绿色标记点 |

---

## 关键设计决策

### 1. DDS 消息类型：复用 `HeightMap_`

用 SDK 已有的 `unitree_go::msg::dds_::HeightMap_`（width/height/data 字段），不需要写 IDL 或生成代码。Topic 名：`rt/depth_image`。

### 2. MuJoCo 深度模拟：`mj_ray()` 而非 `mj_multiRay()`

`mj_ray` 接受 `const mjData*`，线程安全，可从深度发布线程并发调用，不影响物理线程。`mj_multiRay` 接受非 const 指针，会导致栈溢出和数据竞争（5046 条射线一次投射超过 mjData 栈限制 14MB）。

每条射线单独调用 `mj_ray`，5046 条 × 10Hz = ~5万条/秒，CPU 完全够。

### 3. Pinhole 相机模型（对齐 Isaac Lab 训练配置）

```python
# Isaac Lab 训练配置
RayCasterCameraCfg(
    offset=OffsetCfg(pos=(0.3201, 0.0175, 0.08), rot=(1,0,0,0), convention="world"),
    pattern_cfg=PinholeCameraPatternCfg(focal_length=24.0, horizontal_aperture=45.6, width=87, height=58),
    data_types=["distance_to_image_plane"],
    max_distance=2.0,
    depth_clipping_behavior="max",
)
```

对应 MuJoCo 实现：
- 焦距：`fx = focal_length * width / horizontal_aperture = 24*87/45.6 ≈ 45.79 pixels`
- 射线方向（base_link 坐标系，+X=前, -Y=右, +Z=上）：`dir = normalize([1.0, -(u-cx)/fx, (cy-v)/fy])`
- 深度类型：`d_plane = d_euclidean * dir[0]`（`distance_to_image_plane` = 欧几里得距离 × cos θ）
- `bodyexclude=base_link_id`：排除机器人自身几何体

### 4. sim/real 自动切换

```cpp
#ifdef HAS_REALSENSE
    depth_provider_ = std::make_shared<RealSenseDepthCamera>(cam_cfg, env->robot);
#else
    depth_provider_ = std::make_shared<DDSDepthProvider>(env->robot);
#endif
```

- Orin 上编译有 `librealsense2` → `HAS_REALSENSE` 定义 → 真机模式
- PC 上编译无 `librealsense2` → 仿真模式 → DDS 订阅

### 5. 多组观测

`ObservationManager` 原生支持多组观测（检测 YAML 中是否有子组）：

```yaml
observations:
  obs:           # 本体感知组 → ONNX 输入 "obs"
    base_ang_vel: ...
    projected_gravity: ...
    joint_pos_rel: ...
    joint_vel_rel: ...
    velocity_commands: ...
    last_action: ...
  depth:         # 深度组 → ONNX 输入 "depth"
    depth_image:
      params: {width: 87, height: 58}
      scale: [1.0]
      history_length: 1
```

每组生成独立的 `obs_map` 条目，对应 ONNX 模型的不同输入端口。**不需要改任何 C++ 代码**。

---

## 深度预处理参数（Sim2Sim 与 Sim2Real 统一）

| 参数 | Sim2Real (RealSense) | Sim2Sim (MuJoCo) |
|------|---------------------|-------------------|
| 相机位置 (base_link) | `(0.3201, 0.0175, 0.08)` | 同左 |
| 相机朝向 | identity w.r.t. base_link | 同左 |
| 输出分辨率 | 87 × 58 | 87 × 58 |
| 深度范围 | [0.0, 2.0] m | [0.0, 2.0] m |
| 归一化 | [-0.5, 0.5] | [-0.5, 0.5] |
| 深度类型 | `distance_to_image_plane` | `d_euclidean × cos_θ` |
| 无效像素 | 替换为 max_depth (2.0m) | 替换为 max_depth (2.0m) |
| 更新频率 | 10 Hz | 10 Hz |
| 原始分辨率 | 480 × 270 @ 60Hz | 87 × 58 直接生成 |

---

## 校验工具

### test_depth_dds — DDS 深度数据校验

```bash
# 终端1: 启动 MuJoCo
cd unitree_mujoco/simulate/build
./unitree_mujoco -r go2 -s scene_terrain.xml

# 终端2: 校验 DDS 深度
cd unitree_deploy/deploy/robots/go2/build
./test_depth_dds --network lo          # 实时显示 OpenCV 窗口 + 终端统计
./test_depth_dds --network lo --save   # 同时保存 PGM 到 /tmp/depth_dds/
./test_depth_dds --network lo --no-display  # 仅终端统计
```

输出示例：
```
[test_depth_dds] DDS initialized
[test_depth_dds] OpenCV window 'DDS Depth' opened (87x58)
[test_depth_dds] Connected! Receiving depth frames...
[test_depth_dds] frame=50 size=87x58 min=-0.315697 max=0.5 mean=0.119506
```

### MuJoCo 侧 OpenCV 深度图窗口

启动 MuJoCo 时自动弹出 `"Depth Camera (Sim)"` 窗口，显示 MuJoCo 渲染的深度图。内容应与 `test_depth_dds` 窗口完全一致。

### MuJoCo 绿色标记点

`go2.xml` 中新增 `<site name="depth_camera" pos="0.3201 0.0175 0.08" size="0.015" rgba="0 1 0 1"/>`，在机器人前方渲染绿色小球标记深度相机位置。

---

## 配置文件

### unitree_mujoco/simulate/config.yaml

```yaml
depth_camera:
  enable: true
  cam_pos: [0.3201, 0.0175, 0.08]
  cam_quat: [1.0, 0.0, 0.0, 0.0]
  width: 87
  height: 58
  focal_length: 24.0
  horizontal_aperture: 45.6
  min_depth: 0.0
  max_depth: 2.0
  output_min: -0.5
  output_max: 0.5
  update_hz: 10.0
```

### deploy.yaml (策略配置)

```yaml
observations:
  obs:
    base_ang_vel: ...
    # ... 其他本体感知项 ...
  depth:
    depth_image:
      params: {width: 87, height: 58}
      scale: [1.0]
      history_length: 1

depth_camera:
  enable: true
  monitor_only: false
  width: 87
  height: 58
  min_depth: 0.0
  max_depth: 2.0
  output_min: -0.5
  output_max: 0.5
  update_hz: 10.0
```

---

## 涉及的 DDS Topics

| Topic | 消息类型 | 方向 | 频率 | 说明 |
|-------|---------|------|------|------|
| `rt/lowstate` | `LowState_` | MuJoCo→deploy / 固件→deploy | 1 kHz | 关节、IMU（已有） |
| `rt/lowcmd` | `LowCmd_` | deploy→MuJoCo / deploy→固件 | 1 kHz | 电机指令（已有） |
| `rt/depth_image` | `HeightMap_` | MuJoCo→deploy | 10 Hz | **新增** 深度图 |

---

## 线程模型

```
Sim2Sim:
  Physics thread   (mj_step, ~2kHz)
  Bridge thread    (LowState/LowCmd + depth publish, 1kHz)
  Depth pub thread (mj_ray → DDS, 10Hz) ← DepthCameraPublisher内部
  UI thread        (MuJoCo GLFW render)
  Display thread   (OpenCV depth display, ~15Hz)

Sim2Real:
  FSM thread       (LowCmd publish, 1kHz)
  Policy thread    (observation → ONNX → action, 50Hz)
  Camera thread    (RealSense capture → depth_obs, 10Hz)
```

---

## 跨平台兼容

| 平台 | Sim2Sim/Sim2Real | HAS_REALSENSE | 深度数据源 |
|------|-----------------|---------------|-----------|
| x86_64 PC | Sim2Sim | 未定义 | DDS `rt/depth_image` |
| x86_64 PC (带 D435i) | Sim2Sim | 定义 | RealSense D435i |
| aarch64 Jetson Orin | Sim2Real | 定义 | RealSense D435i |

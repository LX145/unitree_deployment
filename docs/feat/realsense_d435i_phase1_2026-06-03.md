# RealSense D435i 深度相机接入 — 第一阶段

**日期**: 2026-06-03

## 目标

在现有盲走策略部署（Sim2Sim / Sim2Real）中，可选地启动 Intel RealSense D435i 深度相机线程，验证相机与控制线程（FSM 1kHz + Policy 50Hz + DDS）的共存性。此阶段深度信息**不接入策略 ONNX**，仅做 monitor_only 模式运行。

## 新增文件

| 文件 | 说明 |
|------|------|
| `deploy/include/sensors/realsense_depth_camera.h` | 深度相机类头文件 |
| `deploy/src/sensors/realsense_depth_camera.cpp` | 深度相机类实现（10Hz 取帧、resize、归一化、debug 保存） |
| `deploy/robots/go2/test_depth_camera.cpp` | Go2 独立测试程序（OpenCV 实时显示 + ASCII 终端回退） |
| `deploy/robots/go2w/test_depth_camera.cpp` | Go2W 独立测试程序 |
| `deploy/robots/g1_29dof/test_depth_camera.cpp` | G1 独立测试程序 |

## 修改文件

| 文件 | 改动说明 |
|------|---------|
| `deploy/include/isaaclab/assets/articulation/articulation.h` | ArticulationData 增加 depth_obs、depth_mtx、depth_valid、depth_timestamp 字段 |
| `deploy/include/FSM/State_RLBase.h` | enter()/exit() 改为声明（移到各机器人 .cpp），新增 `shared_ptr<void> depth_camera_handle_` 成员 |
| `deploy/robots/go2/CMakeLists.txt` | 新增可选 realsense2 支持（pkg-config 检测）、OpenCV 支持、架构自动检测（已有）、test_depth_camera 编译目标 |
| `deploy/robots/go2w/CMakeLists.txt` | 补全 C++17、架构自动检测、可选 realsense2 + OpenCV 支持、test_depth_camera 编译目标 |
| `deploy/robots/g1_29dof/CMakeLists.txt` | 新增可选 realsense2 + OpenCV 支持、test_depth_camera 编译目标 |
| `deploy/robots/go2/src/State_RLBase.cpp` | enter()/exit() 定义 + 构造时读取 deploy.yaml 创建相机（HAS_REALSENSE 保护） |
| `deploy/robots/go2w/src/State_RLBase.cpp` | 同上 |
| `deploy/robots/g1_29dof/src/State_RLBase.cpp` | 同上 |
| `README.md` | 新增 librealsense2-dev 可选依赖说明 |
| `README_zh.md` | 同上 |

## 架构设计

```
RealSenseDepthCamera thread (10 Hz)
  → process_depth() : z16→米→nearest-neighbor resize→clip→normalize
  → history deque (oldest-first)
  → flatten & write to robot->data.depth_obs (mutex-protected)
  → ObservationManager 阶段不读取（留给下阶段）
```

- 深度相机线程与策略线程完全解耦，通过 mutex 保护的共享 buffer 通信
- 使用 `#ifdef HAS_REALSENSE` 宏保护所有相机相关代码，未安装 librealsense2 时自动禁用
- 使用 `pkg-config` 检测 realsense2，避免硬编码路径，兼容 x86_64 和 aarch64
- 退出时调用 `hardware_reset()` 重置相机固件，避免需要物理重插 USB
- 深度预处理全部手写（nearest-neighbor resize），不依赖 OpenCV
- OpenCV 仅用于 test 程序的可视化显示（可选依赖）

## 深度预处理参数

| 参数 | 值 |
|------|-----|
| Raw 分辨率 | 480 × 270 @ 60Hz |
| 输出分辨率 | 87 × 58 |
| 深度范围 | [0.0, 2.0] m |
| 归一化 | [-0.5, 0.5]（近→-0.5, 远→+0.5） |
| 无效像素处理 | 替换为 max_depth (2.0m) |
| 更新频率 | 10 Hz |
| History | 1（单帧） |

## deploy.yaml 配置

```yaml
depth_camera:
  enable: true          # 启用深度相机
  monitor_only: true    # true=相机运行但不接入策略（当前阶段固定为 true）
  save_debug_image: true
  debug_save_dir: "/tmp/depth_debug"
```

## 测试方式

```bash
# 独立相机测试（OpenCV 窗口实时显示灰度深度图）
cd deploy/robots/go2/build
./test_depth_camera

# 与盲走 Sim2Sim 共存测试
# 在 deploy.yaml 中加 depth_camera.enable=true 后正常运行 go2_ctrl
```

## 跨平台兼容

| 平台 | 状态 |
|------|------|
| x86_64 PC (Sim2Sim) | ✅ 已验证 |
| aarch64 Jetson Orin (Sim2Real) | ✅ CMake 自动检测架构，pkg-config 查找依赖 |

## 后续阶段

- 注册 `REGISTER_OBSERVATION(depth_image)` 将深度图接入 Observation 管线
- 部署深度图策略 ONNX（多输入 obs + depth）
- GRU/LSTM recurrent policy 支持

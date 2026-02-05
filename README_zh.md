<div align="center">

[**简体中文**](./README_zh.md) | [**English**](./README.md)

</div>

# Unitree RL Deployment (改进版)

本仓库提供 Unitree 机器人（支持 **G1_29dof**、**Go2** 等）的强化学习（RL）策略部署方案。

### 项目说明
本项目基于 Unitree 官方仓库 [unitree_rl_lab (deploy)](https://github.com/unitreerobotics/unitree_rl_lab/tree/main/deploy) 进行了改进与优化，主要特性包括：
* **多机型支持**：支持 **G1 (29 DOF)** 和 **Go2**。
* **全流程覆盖**：支持从训练完成后的 **Sim2Sim**（Mujoco 仿真验证）到 **Sim2Real**（实机部署）的完整工作流。
* **跨平台兼容**：支持在 **x86 PC**（通过以太网控制）以及 **G1 内部机载电脑 (PC2, ARM64/Jetson Orin)** 上直接进行推理运行。

---

## 1. 环境准备 (Environment Setup)

在标准 PC 上构建部署代码（用于 Sim2Sim 或通过以太网进行 Sim2Real）之前，请先安装以下依赖。

### 1.1 安装系统依赖
```bash
sudo apt update
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

### 1.2 安装 Unitree SDK2
```bash
cd ~
git clone [https://github.com/unitreerobotics/unitree_sdk2.git](https://github.com/unitreerobotics/unitree_sdk2.git)
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF # 安装到 /usr/local 目录
sudo make install
```

---

## 2. 编译控制器 (Build the Controller)

```bash
# 进入对应机器人的目录，例如 G1
cd deploy/robots/g1_29dof 
# 如果是 Go2 机器人，请使用: cd deploy/robots/go2

mkdir build && cd build
cmake .. && make
```

---

## 3. Sim2Sim (Mujoco 仿真)

我们使用 `unitree_mujoco` 进行仿真验证。

### 3.1 配置仿真器
1.  **安装 `unitree_mujoco`**：
    ```bash
    git clone [https://github.com/unitreerobotics/unitree_mujoco.git](https://github.com/unitreerobotics/unitree_mujoco.git)
    cd unitree_mujoco/simulate
    mkdir build && cd build
    cmake .. && make
    ```
2.  **配置 `unitree_mujoco/simulate/config.yaml`**：
    * `robot`: **`g1`** (或 `go2`)
    * `domain_id`: **`0`**
    * `enable_elastic_band`: **`1`** (这对初始站立至关重要)
    * `use_joystick`: **`1`**

### 3.2 运行仿真
**终端 1 (仿真器):**
```bash
cd unitree_mujoco/simulate/build
./unitree_mujoco
```

**终端 2 (控制器):**
```bash
cd deploy/robots/g1_29dof/build
./g1_ctrl
```

### 3.3 操作步骤
1.  **站立 (Stand Up)**：手柄按下 **[L2 + Up]**，机器人将重置为站立姿态。
2.  **落地 (Touch Ground)**：点击 Mujoco 窗口，然后按下键盘 **8**。
3.  **运行策略 (Run Policy)**：手柄按下 **[R1 + X]** 启动 RL 策略。
4.  **释放 (Release)**：点击 Mujoco 窗口，然后按下键盘 **9** 松开弹性绳。

---

## 4. Sim2Real (实机部署)

使用此程序可以通过外部 PC 经由以太网控制机器人。

> **⚠️ 警告**：运行前请确保机器人已**安全吊起**。务必关闭机器人原厂的运动控制服务（Sport Mode）以避免冲突。

```bash
./g1_ctrl --network eth0 # eth0 请替换为您的网卡接口名称
```

---

## 附录：在 G1 机载电脑 (PC2) 上部署

本节介绍如何直接在 G1 的内部电脑（Jetson Orin NX, ARM64/AARCH64）上部署推理策略。

> **注意**：PC2 仅用于推理，请勿用于训练。

### 1. 升级编译器与 CMake
Jetson 默认环境需要更新以支持构建。

**升级 CMake (v3.31.8)**:
```bash
cd ~
wget [https://cmake.org/files/v3.31/cmake-3.31.8.zip](https://cmake.org/files/v3.31/cmake-3.31.8.zip)
unzip cmake-3.31.8.zip && cd cmake-3.31.8
chmod 777 ./configure
./configure
make -j8
sudo make install
sudo update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1 --force
# 验证: cmake --version (应显示 3.31.8)
```

**升级 GCC/G++ (v11)**:
```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt install gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60 --slave /usr/bin/g++ g++ /usr/bin/g++-11
# 如果提示选择，请选择 gcc-11:
sudo update-alternatives --config gcc
```

### 2. 安装 Unitree SDK2 (ARM64)
```bash
cd ~
sudo apt install libeigen3-dev
git clone [https://github.com/unitreerobotics/unitree_sdk2.git](https://github.com/unitreerobotics/unitree_sdk2.git)
cd unitree_sdk2
mkdir build && cd build
cmake ..
make -j8
sudo make install
```

### 3. 安装项目依赖
```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev
```

### 4. 在 PC2 上构建与运行
```bash
cd deploy/robots/g1_29dof
mkdir build && cd build
cmake ..
make -j8

# 运行 (确保机器人已处于调试模式并安全吊起)
./g1_ctrl --network eth0
```

**控制快捷键**:
* `L2 + UP`: 固定站立 (准备就绪)
* `R1 + X`: 启动 RL 策略 (速度控制)
* `LT + LEFT` (按住): 切换至 Mimic 模式 (例如：江南 Style)
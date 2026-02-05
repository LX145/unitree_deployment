<div align="center">

[**简体中文**](./README_zh.md) | [**English**](./README.md)

</div>

# Overview

This project is an improved version based on the official Unitree repository [unitree_rl_lab (deploy)](https://github.com/unitreerobotics/unitree_rl_lab/tree/main/deploy). Key enhancements include:
* **Multi-Robot Support**: Full support for **G1 (29 DOF)** and **Go2**.
* **Complete Workflow**: Covers the entire pipeline from **Sim2Sim** (validation in Mujoco) to **Sim2Real** (physical deployment).
* **Cross-Platform Compatibility**: Supports inference on both **x86 PCs** (via Ethernet control) and the **G1 Onboard Computer (PC2, Jetson Orin)**.

## Setup

### 1. Install Dependencies
```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

### 2. Install unitree_sdk2
```bash
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF # Install on the /usr/local directory
sudo make install
```

### 3. Compile the robot_controller
```bash
# Make sure you are in the unitree_deployment directory
cd deploy/robots/g1_29dof # or other robots like go2
mkdir build && cd build
cmake .. && make
```

## Sim2Sim

We use `unitree_mujoco` for simulation.

1.  **Install `unitree_mujoco`** (if not installed).
2.  Configure the simulation settings in `unitree_mujoco/simulate/config.yaml`:
    * `robot`: **`g1`**
    * `domain_id`: **`0`**
    * `enable_elastic_band`: **`1`**
    * `use_joystick`: **`1`**

### Start Simulation
Open a terminal and run the simulator:
```bash
cd unitree_mujoco/simulate/build
./unitree_mujoco
# Alternative: ./unitree_mujoco -i 0 -n eth0 -r g1 -s scene_29dof.xml
```

### Run Policy
Open a new terminal and run the controller:
```bash
cd unitree_deployment/deploy/robots/g1_29dof/build
./g1_ctrl
```

### Operation Steps
1.  **Stand Up**: Press **[L2 + Up]** on the joystick to set the robot to a standing pose.
2.  **Touch Ground**: Click the Mujoco window, and then press **8** on your keyboard to make the robot feet touch the ground.
3.  **Run Policy**: Press **[R1 + X]** on the joystick to run the policy.
4.  **Release**: Click the Mujoco window, and then press **9** on your keyboard to disable the elastic band.

## Sim2Real

You can use this program to control the robot directly.

> **⚠️ Important:** Make sure the on-board control program has been closed before running this to avoid conflict.

```bash
./g1_ctrl --network eth0 # eth0 is the network interface name
```

# Unitree RL Deployment

This repository contains the deployment code for Unitree robots (G1, Go2, etc.) using Reinforcement Learning policies trained with Isaac Lab.

## 1. Environment Setup (x86 PC)

Before building the deployment code on a standard PC (for Sim2Sim or Sim2Real via Ethernet), install the dependencies.

### 1.1 Install System Dependencies
```bash
sudo apt update
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

### 1.2 Install Unitree SDK2
```bash
cd ~
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF 
sudo make install
```

---

## 2. Build the Controller

```bash
# Navigate to the robot-specific directory (e.g., G1)
cd deploy/robots/g1_29dof 

mkdir build && cd build
cmake .. && make
```

---

## 3. Sim2Sim (Mujoco Simulation)

We use `unitree_mujoco` for simulation.

### 3.1 Setup Simulator
1.  **Install `unitree_mujoco`**:
    ```bash
    git clone https://github.com/unitreerobotics/unitree_mujoco.git
    cd unitree_mujoco/simulate
    mkdir build && cd build
    cmake .. && make
    ```
2.  **Configure `unitree_mujoco/simulate/config.yaml`**:
    * `robot`: **`g1`**
    * `domain_id`: **`0`**
    * `enable_elastic_band`: **`1`** (Crucial for initial standing)
    * `use_joystick`: **`1`**

### 3.2 Run Simulation
**Terminal 1 (Simulator):**
```bash
cd unitree_mujoco/simulate/build
./unitree_mujoco
```

**Terminal 2 (Controller):**
```bash
cd deploy/robots/g1_29dof/build
./g1_ctrl
```

### 3.3 Operation
1.  **Stand Up**: Press **[L2 + Up]** (Joystick) -> Robot resets to standing.
2.  **Touch Ground**: Click Mujoco window -> Press **8** (Keyboard).
3.  **Run Policy**: Press **[R1 + X]** (Joystick).
4.  **Release**: Click Mujoco window -> Press **9** (Keyboard).

---

## 4. Sim2Real (Standard)

Use this to run the policy on a robot from an external PC via Ethernet.

> **⚠️ Warning**: Ensure the robot is suspended safely. Close the default sport mode service on the robot before running.

```bash
./g1_ctrl --network eth0
```

---

## Appendix: Deployment on G1 Onboard Computer (PC2)

This section describes how to deploy the inference policy directly on G1's onboard computer (Jetson Orin NX, ARM64/AARCH64).

> **Note**: PC2 is used for inference only. Do not use it for training.

### 1. Upgrade Compiler & CMake
The default Jetson environment requires updates to support the build.

**Upgrade CMake (v3.31.8)**:
```bash
cd ~
wget https://cmake.org/files/v3.31/cmake-3.31.8.zip
unzip cmake-3.31.8.zip && cd cmake-3.31.8
chmod 777 ./configure
./configure
make -j8
sudo make install
sudo update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1 --force
# Verify: cmake --version (Should be 3.31.8)
```

**Upgrade GCC/G++ (v11)**:
```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt install gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60 --slave /usr/bin/g++ g++ /usr/bin/g++-11
# Choose gcc-11 if prompted:
sudo update-alternatives --config gcc
```

### 2. Install Unitree SDK2 (ARM64)
```bash
cd ~
sudo apt install libeigen3-dev
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake ..
make -j8
sudo make install
```

### 3. Install Project Dependencies
```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev
```

### 4. Build & Run on PC2
```bash
cd deploy/robots/g1_29dof
mkdir build && cd build
cmake ..
make -j8

# Run (Ensure robot is suspended and in debug mode)
./g1_ctrl --network eth0
```

**Controls**:
* `L2 + UP`: FixStand (Ready)
* `R1 + X`: Start RL Policy (Velocity)
* `LT + LEFT` (Hold): Switch to Mimic Mode (Gangnam Style)
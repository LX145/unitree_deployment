# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deployment system for Unitree robots (G1 humanoid 29-DOF, Go2 quadruped, Go2W wheeled) running Reinforcement Learning policies trained in Nvidia Isaac Lab. Policies are exported as ONNX models and executed via ONNX Runtime. Supports both Sim2Sim (MuJoCo) and Sim2Real (physical robot via Ethernet or onboard Jetson Orin).

## Build

```bash
cd deploy/robots/g1_29dof   # or go2, go2w
mkdir build && cd build
cmake .. && make -j$(nproc)
```

CMake auto-detects `aarch64` (Jetson) vs `x86_64` (PC) and selects the matching ONNX Runtime binary from `deploy/thirdparty/`. C++17 required.

**System dependencies** (must be pre-installed):
`libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev unitree_sdk2`

`unitree_sdk2` must be built from source: `cmake .. -DBUILD_EXAMPLES=OFF && sudo make install`.

No test suite exists in this repository.

## Run

```bash
# Sim2Sim (MuJoCo) — start simulator first, then:
./g1_ctrl                        # local simulation

# Sim2Real (physical robot):
./g1_ctrl --network eth0         # specify DDS network interface
./g1_ctrl --log                  # enable file logging to log/log.txt
```

CLI flags: `--help`, `--version`, `--log`, `--network`/`-n` (default: empty string).

## Architecture

### Three-Layer Structure

```
deploy/
  include/           Shared headers (all robots)
    FSM/             Finite State Machine framework
    isaaclab/        Isaac Lab RL environment (observation/action managers, articulation)
    param.h          Config loading + CLI parsing
    unitree_joystick_dsl.hpp  Custom DSL for joystick→FSM transition rules
  robots/
    g1_29dof/        G1 humanoid (29 DOF) — the primary target
    go2/             Go2 quadruped
    go2w/            Go2W wheeled variant
  thirdparty/        ONNX Runtime 1.22.0 binaries (x86_64 + aarch64)
```

Each robot directory has the same shape: `main.cpp`, `CMakeLists.txt`, `config/config.yaml`, `config/policy/*/params/deploy.yaml`, `config/policy/*/exported/policy.onnx`, `include/Types.h`, `src/State_RLBase.cpp`.

### Execution Flow

1. `main.cpp` calls `param::helper()` → loads `config.yaml` from `../config/` relative to the binary, parses CLI args
2. Initializes Unitree DDS channel (`ChannelFactory::Instance()->Init()`)
3. `init_fsm_state()` claims the low-level command channel, connects to robot LowState
4. Creates a `CtrlFSM` from the `FSM` section of config.yaml
5. `fsm->start()` launches a **1 kHz recurrent thread** running the FSM loop
6. Main thread sleeps indefinitely

### FSM Framework (`deploy/include/FSM/`)

- `CtrlFSM` — owns the state list, runs the 1kHz loop, checks transitions each cycle
- `FSMState` — base for all states; at construction, parses `transitions` from config.yaml using the joystick DSL and registers check→target_state pairs. Also registers a global timeout→Passive fallback. Static members `lowcmd` and `lowstate` are shared across all states.
- `BaseState` — abstract interface: `enter()`, `run()`, `exit()`, `pre_run()`, `post_run()`
- Concrete states: `State_Passive` (damping only), `State_FixStand` (interpolates to standing pose), `State_RLBase` (runs ONNX policy), `State_Mimic` (G1 only, BVH motion playback)

States register themselves via `REGISTER_FSM(StateName)` macro so `CtrlFSM` can instantiate them by name from config.

### Joystick DSL (`unitree_joystick_dsl.hpp`)

A custom lexer/parser/compiler that turns human-readable transition conditions in config.yaml into executable predicates. Supports:

- Button names: `A`, `B`, `X`, `Y`, `LB`, `RB`, `LT`, `RT`, `up`, `down`, `left`, `right`, `start`, `back`, `LX`, `LY`, `RX`, `RY`
- States: `.pressed` (held), `.on_pressed` (single-frame edge), `.on_released`
- Long press: `LT(2s)` — held ≥ N seconds
- Operators: `+` (AND), `|` (OR), `!` (NOT), `()` grouping
- Example from config: `LT + up.on_pressed` (LT held + up just pressed), `LT(2s) + left.on_pressed`

The DSL is compiled at FSMState construction time into `std::function<bool(UnitreeJoystick)>` predicates. These are checked every 1kHz cycle in `CtrlFSM::run_()`.

### Isaac Lab RL Environment (`deploy/include/isaaclab/`)

`ManagerBasedRLEnv` is the core RL inference loop, mirroring Isaac Lab's Python API in C++:

- `ObservationManager` — concatenates observation terms (base angular velocity, projected gravity, joint positions/velocities, velocity commands, last action) into an input tensor, applying per-term scaling and clipping
- `ActionManager` — applies scale/offset/clip to raw policy output, producing joint position targets
- `Articulation` — abstract interface for reading robot state; `unitree::BaseArticulation<LowStatePtr>` implements it by reading from DDS LowState messages (IMU, joint positions/velocities)
- `Algorithms` — wraps ONNX Runtime inference via `OrtRunner`
- `MotionCommand` — generates velocity command inputs (base linear/angular velocity targets)

The RL policy runs in a **separate thread** at `step_dt` Hz (typically 0.02s = 50Hz), while the FSM runs at 1kHz for joint-level PD control.

### Configuration

**`config.yaml`** — defines which FSM states are enabled, their numeric IDs, transition rules (joystick DSL expressions), and per-state parameters:
- `Passive`: damping gains only
- `FixStand`: PD gains (`kp`, `kd`), interpolation times `ts` and target joint positions `qs`
- `Velocity` (type `RLBase`): `policy_dir` path
- `Mimic_*` (type `Mimic`): `motion_file` (BVH CSV), `fps`, `policy_dir`

**`deploy.yaml`** (per-policy directory) — defines RL environment parameters: `step_dt`, `joint_ids_map`, `stiffness`/`damping`, `default_joint_pos`, action config (`scale`, `offset`, `clip`), observation terms (each with `scale`, `clip`, `history_length`), and command ranges.

### Robot-Specific Types

Each robot defines its own `Types.h` with `LowCmd_t` and `LowState_t` typedefs that wrap the appropriate Unitree SDK channel types (e.g., `unitree::robot::g1::subscription::LowState`). The shared `FSMState` uses these via static members, allowing the FSM framework to be robot-agnostic.

### Adding a New Policy

1. Place `policy.onnx` and `params/deploy.yaml` under `config/policy/<category>/<name>/`
2. Reference the directory via `policy_dir` in config.yaml under a Velocity or Mimic state
3. The `param::parser_policy_dir()` helper auto-resolves relative paths and finds the latest versioned subdirectory if `exported/` isn't directly present

### Adding a New Robot

1. Create `deploy/robots/<name>/` with the standard structure (`main.cpp`, `CMakeLists.txt`, `include/Types.h`, `src/State_RLBase.cpp`, `config/config.yaml`)
2. Define `LowCmd_t` and `LowState_t` wrapping the correct Unitree SDK channel types
3. The shared FSM and IsaacLab code is robot-agnostic — only the DDS channel types and joint counts differ

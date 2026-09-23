// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include <cstdio>
#include <vector>
#include <string>
#include <memory>
#include <atomic>
#include <functional>
#include <mutex>

class DepthProvider;  // forward declaration

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);

    bool can_enter() override;

    void enter() override;

    void run();

    void exit() override;

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    std::atomic<bool> policy_thread_running{false};
    std::atomic<bool> policy_action_ready_{false};

    FILE* log_file = nullptr;       // 文件指针
    char write_buffer[1024 * 1024]; // 1MB 的写缓冲区，避免频繁触发磁盘 I/O
    long long log_step_count = 0;   // 用于生成时间戳

    // Depth provider: RealSense (real) or DDS (sim), both write to env->robot->data.depth_obs
    std::shared_ptr<DepthProvider> depth_provider_;
    std::vector<float> entry_joint_pos_;
    double policy_action_warmup_s_ = 0.0;
    bool rl_gains_applied_ = false;

    bool timing_log_enabled_ = false;
    bool timing_log_autostart_ = true;
    bool timing_log_active_ = false;
    bool timing_log_started_once_ = false;
    bool timing_log_toggle_latched_ = false;
    std::function<bool(const unitree::common::UnitreeJoystick&)> timing_log_toggle_check_;
    std::string timing_log_path_;
    FILE* timing_log_file_ = nullptr;
    char timing_log_buffer_[256 * 1024];
    std::mutex timing_log_mtx_;

    void open_timing_log();
    void close_timing_log();
};

REGISTER_FSM(State_RLBase)

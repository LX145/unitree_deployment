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
    bool policy_thread_running = false;

    FILE* log_file = nullptr;       // 文件指针
    char write_buffer[1024 * 1024]; // 1MB 的写缓冲区，避免频繁触发磁盘 I/O
    long long log_step_count = 0;   // 用于生成时间戳

    // Depth provider: RealSense (real) or DDS (sim), both write to env->robot->data.depth_obs
    std::shared_ptr<DepthProvider> depth_provider_;
    std::vector<float> entry_joint_pos_;
};

REGISTER_FSM(State_RLBase)

// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/assets/articulation/articulation.h"
#include <chrono>

namespace unitree
{

template <typename LowStatePtr>
class BaseArticulation : public isaaclab::Articulation
{
public:
    BaseArticulation(LowStatePtr lowstate_)
    : lowstate(lowstate_)
    {
        data.joystick = &lowstate->joystick;
    }

    void update() override
    {
        const auto sample_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        data.lowstate_tick = lowstate->msg_.tick();
        data.lowstate_sample_timestamp = sample_time;
        // base_angular_velocity
        for(int i(0); i<3; i++) {
            data.root_ang_vel_b[i] = lowstate->msg_.imu_state().gyroscope()[i];
        }
        for(int i(0); i<3; i++) {
            data.imu_acc[i] = lowstate->msg_.imu_state().accelerometer()[i];
        }
        // project_gravity_body
        data.root_quat_w = Eigen::Quaternionf(
            lowstate->msg_.imu_state().quaternion()[0],
            lowstate->msg_.imu_state().quaternion()[1],
            lowstate->msg_.imu_state().quaternion()[2],
            lowstate->msg_.imu_state().quaternion()[3]
        );
        data.projected_gravity_b = data.root_quat_w.conjugate() * data.GRAVITY_VEC_W;
        // joint positions and velocities
        for(int i(0); i< data.joint_ids_map.size(); i++) {
            data.joint_pos[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].q();
            data.joint_vel[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].dq();
        }
    }

    LowStatePtr lowstate;
};

}
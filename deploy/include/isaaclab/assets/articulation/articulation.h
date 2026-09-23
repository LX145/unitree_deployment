// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include "unitree/dds_wrapper/common/unitree_joystick.hpp"
#include <mutex>
#include <vector>
#include <cstdint>

namespace isaaclab
{

class MotionLoader;

struct ArticulationData
{
    Eigen::Vector3f GRAVITY_VEC_W = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
    Eigen::Vector3f FORWARD_VEC_B = Eigen::Vector3f(1.0f, 0.0f, 0.0f);

    std::vector<float> joint_stiffness; // sdk order
    std::vector<float> joint_damping; // sdk order

    Eigen::Vector3f imu_acc = Eigen::Vector3f::Zero();

    // Joint positions of all joints.
    Eigen::VectorXf joint_pos;

    // Default joint positions of all joints.
    Eigen::VectorXf default_joint_pos;

    // Joint velocities of all joints.
    Eigen::VectorXf joint_vel;

    // Root angular velocity in base world frame.
    Eigen::Vector3f root_ang_vel_b;

    // Projection of the gravity direction on base frame.
    Eigen::Vector3f projected_gravity_b;

    Eigen::Quaternionf root_quat_w;

    std::vector<float> joint_ids_map;

    unitree::common::UnitreeJoystick* joystick = nullptr;

    isaaclab::MotionLoader* motion_loader = nullptr;

    // ---- depth camera buffer (mutex-protected) ----
    // Preprocessed, normalized, history-stacked, flattened policy input.
    // Written by RealSenseDepthCamera or DDS thread at ~50 Hz.
    // Read by observation pipeline at ~50 Hz.
    std::vector<float> depth_obs;
    mutable std::mutex depth_mtx;
    bool depth_valid = false;
    // Legacy alias for the local receive/write time. Prefer the explicit fields below.
    double depth_timestamp = 0.0;
    double depth_source_timestamp = 0.0;  // Sensor/simulator-provided stamp, time base may differ.
    double depth_rx_timestamp = 0.0;      // Local steady-clock time when depth_obs was written.
    uint64_t depth_frame_number = 0;      // Sensor/simulator frame counter if available.
    uint64_t depth_seq = 0;               // Monotonically increasing local depth buffer counter.

    // Metadata for the exact depth buffer read by the observation pipeline.
    bool depth_obs_last_read_valid = false;
    uint64_t depth_obs_last_read_seq = 0;
    uint64_t depth_obs_last_read_frame_number = 0;
    double depth_obs_last_read_source_timestamp = 0.0;
    double depth_obs_last_read_rx_timestamp = 0.0;

    // Metadata for the latest lowstate sampled by Articulation::update().
    uint32_t lowstate_tick = 0;
    double lowstate_sample_timestamp = 0.0;
};

class Articulation
{
public:
    Articulation(){}

    virtual void update(){};

    ArticulationData data;
};

};

// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

/**
 * @brief Abstract interface for depth data providers.
 *
 * RealSenseDepthCamera (real hardware) and DDSDepthProvider (simulation)
 * both implement this interface, allowing the policy code to consume depth
 * data from a unified source without knowing whether it's sim or real.
 */
class DepthProvider {
public:
    virtual ~DepthProvider() = default;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual bool is_running() const = 0;
    virtual bool is_available() const = 0;
    virtual bool is_ready() const = 0;
    virtual bool has_failed() const = 0;
};

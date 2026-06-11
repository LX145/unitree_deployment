// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <unitree/dds_wrapper/common/Subscription.h>
#include <unitree/idl/go2/HeightMap_.hpp>

#include <isaaclab/assets/articulation/articulation.h>
#include "sensors/depth_provider.h"

#include <memory>
#include <atomic>
#include <spdlog/spdlog.h>

/**
 * @brief Receives depth frames from MuJoCo simulation via DDS and writes them
 *        into ArticulationData::depth_obs, mirroring what RealSenseDepthCamera
 *        does on real hardware.
 *
 * Usage (simulation mode):
 *   auto provider = std::make_shared<DDSDepthProvider>(robot);
 *   provider->start();   // begins subscribing to rt/depth_image
 *   // ... policy runs, reads robot->data.depth_obs ...
 *   provider->stop();
 */
class DDSDepthProvider : public DepthProvider {
public:
    using HeightMap_t = unitree_go::msg::dds_::HeightMap_;

    explicit DDSDepthProvider(std::shared_ptr<isaaclab::Articulation> robot)
        : robot_(std::move(robot))
    {
        if (!robot_) {
            throw std::runtime_error("DDSDepthProvider: robot must not be null");
        }
    }

    ~DDSDepthProvider() { stop(); }

    /// Start subscribing to rt/depth_image. Idempotent.
    void start()
    {
        if (running_.load()) return;

        sub_ = std::make_shared<unitree::robot::SubscriptionBase<HeightMap_t>>(
            "rt/depth_image",
            [this](const void* msg) {
                const auto& heightmap = *static_cast<const HeightMap_t*>(msg);
                std::lock_guard<std::mutex> lock(robot_->data.depth_mtx);

                robot_->data.depth_obs = heightmap.data();
                robot_->data.depth_valid = !heightmap.data().empty();
                robot_->data.depth_timestamp = heightmap.stamp();
                robot_->data.depth_seq++;
            });

        // Wait for publisher to come online (with timeout)
        sub_->set_timeout_ms(5000);
        sub_->wait_for_connection();

        running_.store(true);
        spdlog::info("[DDSDepth] subscribed to rt/depth_image");
    }

    /// Stop subscription. Idempotent.
    void stop()
    {
        if (!running_.load()) return;
        running_.store(false);
        sub_.reset();
        spdlog::info("[DDSDepth] stopped");
    }

    bool is_running() const { return running_.load(); }

private:
    std::shared_ptr<isaaclab::Articulation> robot_;
    std::shared_ptr<unitree::robot::SubscriptionBase<HeightMap_t>> sub_;
    std::atomic<bool> running_{false};
};

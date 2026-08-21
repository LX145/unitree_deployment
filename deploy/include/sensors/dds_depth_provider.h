// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <unitree/dds_wrapper/common/Subscription.h>
#include <unitree/idl/go2/HeightMap_.hpp>

#include <isaaclab/assets/articulation/articulation.h>
#include "sensors/depth_provider.h"

#include <memory>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

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

    DDSDepthProvider(std::shared_ptr<isaaclab::Articulation> robot,
                     const YAML::Node& cfg)
        : robot_(std::move(robot))
        , log_distribution_(cfg["log_distribution"].as<bool>(false))
        , width_(cfg["width"].as<int>(87))
        , height_(cfg["height"].as<int>(58))
        , output_max_(cfg["output_max"].as<float>(0.5f))
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
        ready_.store(false);

        sub_ = std::make_shared<unitree::robot::SubscriptionBase<HeightMap_t>>(
            "rt/depth_image",
            [this](const void* msg) {
                const auto& heightmap = *static_cast<const HeightMap_t*>(msg);
                const bool valid = !heightmap.data().empty();
                {
                    std::lock_guard<std::mutex> lock(robot_->data.depth_mtx);
                    robot_->data.depth_obs = heightmap.data();
                    robot_->data.depth_valid = valid;
                    robot_->data.depth_timestamp = heightmap.stamp();
                    robot_->data.depth_seq++;
                }
                ready_.store(valid);
                log_distribution(heightmap.data());
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
        ready_.store(false);
        sub_.reset();
        spdlog::info("[DDSDepth] stopped");
    }

    bool is_running() const override { return running_.load(); }
    bool is_ready() const override { return ready_.load(); }
    bool has_failed() const override { return false; }

private:
    void log_distribution(const std::vector<float>& depth_obs)
    {
        if (!log_distribution_ ||
            depth_obs.size() != static_cast<std::size_t>(width_ * height_)) return;

        const auto now = std::chrono::steady_clock::now();
        if (last_distribution_log_time_.time_since_epoch().count() != 0 &&
            now - last_distribution_log_time_ < std::chrono::seconds(1)) return;
        last_distribution_log_time_ = now;

        const float saturated_threshold = output_max_ - 1.0e-6f;
        const auto saturated_count = std::count_if(
            depth_obs.begin(), depth_obs.end(),
            [saturated_threshold](float value) { return value >= saturated_threshold; });
        const double saturated_percent =
            100.0 * static_cast<double>(saturated_count) / depth_obs.size();
        const double mean = std::accumulate(depth_obs.begin(), depth_obs.end(), 0.0)
                          / depth_obs.size();

        std::ostringstream row_profile;
        row_profile << std::fixed << std::setprecision(3) << '[';
        for (int y = 0; y < height_; y += 2) {
            const int row_end = std::min(y + 2, height_);
            double sum = 0.0;
            for (int row = y; row < row_end; ++row) {
                sum += std::accumulate(
                    depth_obs.begin() + row * width_,
                    depth_obs.begin() + (row + 1) * width_,
                    0.0);
            }
            if (y != 0) row_profile << ", ";
            row_profile << sum / (static_cast<double>(row_end - y) * width_);
        }
        row_profile << ']';

        spdlog::info(
            "[Depth Stats][MuJoCo DDS] mean={:.3f} max_saturated={}/{} ({:.1f}%) "
            "row_pair_mean(top->bottom)={}",
            mean, saturated_count, depth_obs.size(), saturated_percent,
            row_profile.str());
    }

    std::shared_ptr<isaaclab::Articulation> robot_;
    std::shared_ptr<unitree::robot::SubscriptionBase<HeightMap_t>> sub_;
    std::atomic<bool> running_{false};
    std::atomic<bool> ready_{false};
    bool log_distribution_ = false;
    int width_ = 87;
    int height_ = 58;
    float output_max_ = 0.5f;
    std::chrono::steady_clock::time_point last_distribution_log_time_{};
};

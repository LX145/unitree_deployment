// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <atomic>
#include <thread>
#include <yaml-cpp/yaml.h>

#include "sensors/depth_provider.h"

namespace isaaclab {
class Articulation;
}

/**
 * @brief RealSense D435i depth camera wrapper for RL policy deployment.
 *
 * Runs a background thread that captures depth frames at configurable Hz,
 * preprocesses them (resize, clip, normalize), and writes the result into
 * ArticulationData::depth_obs under a mutex for consumption by the policy.
 *
 * Usage:
 *   auto cam = RealSenseDepthCamera(cfg, robot);
 *   cam.start();   // spawns background thread
 *   // ... policy runs, reads robot->data.depth_obs ...
 *   cam.stop();    // joins thread, stops pipeline
 */
class RealSenseDepthCamera : public DepthProvider {
public:
    struct Config {
        // ---- control ----
        bool enable = false;
        bool monitor_only = true;   // true = camera runs but policy ignores depth
        std::string fail_behavior = "warn_only";  // "warn_only" | "passive"

        // ---- raw camera ----
        int raw_width = 480;
        int raw_height = 270;
        int raw_fps = 60;

        // ---- output (policy input) ----
        int out_width = 87;
        int out_height = 58;
        int history = 1;
        float update_hz = 10.0f;

        // ---- depth normalization ----
        float min_depth = 0.0f;
        float max_depth = 2.0f;
        float output_min = -0.5f;
        float output_max = 0.5f;

        // ---- processing ----
        bool replace_invalid_with_max = true;

        // ---- debug ----
        bool save_debug_image = false;
        float debug_save_interval_s = 2.0f;
        std::string debug_save_dir = "/tmp/depth_debug";

        /// Load config from a YAML node (typically deploy.yaml's "depth_camera" section).
        static Config from_yaml(const YAML::Node& node);
    };

    /**
     * @param cfg  Camera configuration.
     * @param robot  Articulation whose data.depth_obs will be written to.
     */
    RealSenseDepthCamera(const Config& cfg,
                         std::shared_ptr<isaaclab::Articulation> robot);
    ~RealSenseDepthCamera();

    /// Start the background capture thread.  Idempotent.
    void start();

    /// Stop the background capture thread and close the pipeline.  Idempotent.
    void stop();

    /// Returns true if the capture thread is currently running.
    bool is_running() const override { return running_.load(); }
    bool is_ready() const override { return ready_.load(); }
    bool has_failed() const override { return failed_.load(); }

private:
    /// The background loop: capture → preprocess → write to robot->data.
    void capture_loop();

    /// Process a single raw depth frame (z16) into a normalized, resized
    /// flat vector of length out_width * out_height.  Returned values are
    /// in [output_min, output_max].
    std::vector<float> process_depth(const uint16_t* raw,
                                     int raw_w, int raw_h,
                                     float depth_scale);

    /// Save the normalized depth image to disk (PGM text format + raw binary).
    void save_debug_frame(const std::vector<float>& frame);

    Config cfg_;
    std::shared_ptr<isaaclab::Articulation> robot_;

    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> ready_{false};
    std::atomic<bool> failed_{false};

    // ---- statistics ----
    int64_t processed_frame_count_ = 0;
    double last_save_time_ = 0.0;
};

// helper: get monotonic time in seconds
double now_sec();

/**
 * @file test_depth_camera.cpp
 * @brief Standalone test: read and preprocess D435i depth frames without
 *        connecting to a robot.  Prints statistics and optionally saves
 *        debug images.
 *
 * Build:  cd deploy/robots/g1_29dof/build && cmake .. && make test_depth_camera
 * Usage:  ./test_depth_camera
 */

#include "sensors/realsense_depth_camera.h"
#include "isaaclab/assets/articulation/articulation.h"

#include <iostream>
#include <csignal>
#include <numeric>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

volatile sig_atomic_t g_stop = 0;

void on_signal(int) { g_stop = 1; }

int main()
{
    // ----- configure logger -----
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%H:%M:%S] [%^%l%$] %v");

    // ----- configure camera -----
    RealSenseDepthCamera::Config cfg;
    cfg.enable       = true;
    cfg.monitor_only = true;
    cfg.raw_width    = 480;
    cfg.raw_height   = 270;
    cfg.raw_fps      = 60;
    cfg.out_width    = 87;
    cfg.out_height   = 58;
    cfg.history      = 1;
    cfg.update_hz    = 10.0f;
    cfg.min_depth    = 0.0f;
    cfg.max_depth    = 2.0f;
    cfg.output_min   = -0.5f;
    cfg.output_max   = 0.5f;
    cfg.replace_invalid_with_max = true;
    cfg.save_debug_image         = true;
    cfg.debug_save_interval_s    = 2.0f;
    cfg.debug_save_dir           = "/tmp/depth_debug";

    // ---- optionally load overrides from YAML ----
    try {
        YAML::Node yaml = YAML::LoadFile("test_depth_cfg.yaml");
        if (yaml) cfg = RealSenseDepthCamera::Config::from_yaml(yaml);
    } catch (...) {
        spdlog::info("No test_depth_cfg.yaml found, using defaults");
    }

    // ---- create a dummy articulation (just for data buffer) ----
    auto robot = std::make_shared<isaaclab::Articulation>();

    // ---- create & start camera ----
    RealSenseDepthCamera cam(cfg, robot);
    cam.start();

    spdlog::info("Depth camera running.  Press Ctrl+C to stop.");
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    // ---- idle loop ----
    while (!g_stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    spdlog::info("Shutting down...");
    cam.stop();

    // ---- final stats ----
    {
        std::lock_guard<std::mutex> lock(robot->data.depth_mtx);
        if (robot->data.depth_valid && !robot->data.depth_obs.empty()) {
            auto& obs = robot->data.depth_obs;
            auto [mn, mx] = std::minmax_element(obs.begin(), obs.end());
            float mean = std::accumulate(obs.begin(), obs.end(), 0.0f) / obs.size();
            spdlog::info("Final depth_obs: size={} min={:.4f} max={:.4f} mean={:.4f}",
                         obs.size(), *mn, *mx, mean);
        } else {
            spdlog::warn("No valid depth frame received");
        }
    }

    spdlog::info("Done.");
    return 0;
}

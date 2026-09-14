/**
 * @file test_depth_camera.cpp
 * @brief Standalone D435i depth camera test with OpenCV display.
 *
 * Build:  cd deploy/robots/go2/build && cmake .. && make test_depth_camera
 * Usage:  ./test_depth_camera
 */

#include "sensors/realsense_depth_camera.h"
#include "isaaclab/assets/articulation/articulation.h"

#include <iostream>
#include <iomanip>
#include <csignal>
#include <filesystem>
#include <numeric>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#ifdef HAS_OPENCV
#include <opencv2/opencv.hpp>
#endif

volatile sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

#ifdef HAS_OPENCV
// ---------------------------------------------------------------------------
// OpenCV depth visualisation
// ---------------------------------------------------------------------------
static void show_depth_opencv(const std::vector<float>& depth_obs,
                              int w, int h, int frame_count,
                              float output_min, float output_max,
                              float min_depth, float max_depth)
{
    const float output_range = output_max - output_min;
    cv::Mat gray(h, w, CV_32FC1);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float value = depth_obs[y * w + x];
            gray.at<float>(y, x) = std::clamp(
                (output_max - value) / output_range, 0.0f, 1.0f);
        }
    }
    gray.convertTo(gray, CV_8UC1, 255.0);

    // 4x nearest-neighbour for visibility
    cv::Mat big;
    cv::resize(gray, big, cv::Size(), 4.0, 4.0, cv::INTER_NEAREST);

    // Convert to BGR for coloured overlay text
    cv::Mat display;
    cv::cvtColor(big, display, cv::COLOR_GRAY2BGR);

    // Overlay text (white)
    char buf[128];
    snprintf(buf, sizeof(buf), "#%d  %dx%d  [white=%.2fm black=%.2fm]",
             frame_count, w, h, min_depth, max_depth);
    cv::putText(display, buf, cv::Point(4, 12),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(200, 200, 200), 1);

    // Scale bar at bottom
    int bar_y = big.rows - 12;
    int bar_w = big.cols;
    for (int x = 0; x < bar_w; ++x) {
        uchar v = static_cast<uchar>(255 - (x * 255 / bar_w));  // left=near(white), right=far(black)
        cv::line(display, cv::Point(x, bar_y), cv::Point(x, bar_y + 8),
                 cv::Scalar(v, v, v), 1);
    }
    snprintf(buf, sizeof(buf), "%.2fm", min_depth);
    cv::putText(display, buf, cv::Point(2, bar_y - 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(200, 200, 200), 1);
    snprintf(buf, sizeof(buf), "%.2fm", max_depth);
    cv::putText(display, buf, cv::Point(bar_w - 34, bar_y - 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(200, 200, 200), 1);

    cv::imshow("RealSense D435i Depth", display);
}

#else
// ---------------------------------------------------------------------------
// Fallback: simple terminal output (no OpenCV)
// ---------------------------------------------------------------------------
static void show_depth_terminal(const std::vector<float>& depth_obs,
                                int w, int h,
                                double timestamp, int frame_count,
                                float output_min, float output_max)
{
    std::cout << "\033[2J\033[H" << std::flush;  // clear screen
    std::cout << "=== D435i Depth  frame=" << frame_count
              << "  " << w << "x" << h
              << "  age=" << std::fixed << std::setprecision(0)
              << (now_sec() - timestamp) * 1000.0 << "ms  (Ctrl+C) ===\n";

    // Simple ASCII with half-blocks and ANSI colors
    const char* ramp = " .:-=+*#%@";
    const float output_range = output_max - output_min;
    for (int y = 0; y < h; y += 2) {
        for (int x = 0; x < w; ++x) {
            const float value = depth_obs[y * w + x];
            int idx = static_cast<int>((output_max - value) / output_range * 10.0f);
            idx = std::clamp(idx, 0, 9);
            std::cout << ramp[idx];
        }
        std::cout << '\n';
    }
    std::cout << std::flush;
}
#endif

// ---------------------------------------------------------------------------
int main()
{
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%H:%M:%S] [%^%l%$] %v");

    const auto deploy_config_path =
        std::filesystem::path(GO2_SOURCE_DIR) /
        "config/policy/velocity/depth_student/params/deploy.yaml";
    YAML::Node deploy_cfg;
    try {
        deploy_cfg = YAML::LoadFile(deploy_config_path.string());
    } catch (const YAML::Exception& error) {
        spdlog::error("Failed to load {}: {}", deploy_config_path.string(), error.what());
        return 1;
    }
    if (!deploy_cfg["depth_camera"]) {
        spdlog::error("{} has no depth_camera section", deploy_config_path.string());
        return 1;
    }
    auto cfg = RealSenseDepthCamera::Config::from_yaml(deploy_cfg["depth_camera"]);
    cfg.monitor_only = true;
    // This standalone process has no controller DDS initialization. Preview the
    // local policy-input buffer directly instead of republishing it.
    cfg.publish_debug_dds = false;
    spdlog::info(
        "Loaded deployment depth pipeline from {}: raw={}x{}@{}Hz, output={}x{}@{:.1f}Hz, "
        "depth=[{:.2f}, {:.2f}]m, invalid_below={:.2f}m",
        deploy_config_path.string(), cfg.raw_width, cfg.raw_height, cfg.raw_fps,
        cfg.out_width, cfg.out_height, cfg.update_hz,
        cfg.min_depth, cfg.max_depth, cfg.invalid_depth_threshold);

    auto robot = std::make_shared<isaaclab::Articulation>();

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    RealSenseDepthCamera cam(cfg, robot);
    cam.start();

    spdlog::info("Depth camera starting; waiting for first valid frame...");

#ifdef HAS_OPENCV
    bool window_created = false;
#endif

    int frame_count = 0;
    bool camera_failed = false;
    while (!g_stop) {
        if (cam.has_failed()) {
            spdlog::error("Depth camera test failed: pipeline did not produce a valid stream");
            camera_failed = true;
            break;
        }

        // Read latest depth from shared buffer
        std::vector<float> frame;
        double ts = 0;
        {
            std::lock_guard<std::mutex> lock(robot->data.depth_mtx);
            if (robot->data.depth_valid && !robot->data.depth_obs.empty()) {
                frame = robot->data.depth_obs;
                ts = robot->data.depth_timestamp;
            }
        }

        if (!frame.empty()) {
#ifdef HAS_OPENCV
            if (!window_created) {
                cv::namedWindow("RealSense D435i Depth", cv::WINDOW_AUTOSIZE);
                window_created = true;
                spdlog::info("First valid depth frame received. Press Ctrl+C, Q, or ESC to stop.");
            }
            show_depth_opencv(frame, cfg.out_width, cfg.out_height,
                              frame_count, cfg.output_min, cfg.output_max,
                              cfg.min_depth, cfg.max_depth);

            const int display_delay_ms = std::max(
                1, static_cast<int>(std::lround(1000.0f / cfg.update_hz)));
            const int key = cv::waitKey(display_delay_ms);
            const int key_code = key < 0 ? key : key & 0xff;
            if (key_code == 3 || key_code == 27 || key_code == 'q' || key_code == 'Q') {
                // Ctrl+C is delivered as ASCII ETX (3) when the OpenCV window,
                // rather than the launching terminal, owns keyboard focus.
                spdlog::info("Key pressed, exiting");
                g_stop = 1;
            }
            // NOTE: do NOT auto-detect window close via getWindowProperty —
            // it returns -1 on some systems even when the window is visible.
#else
            show_depth_terminal(frame, cfg.out_width, cfg.out_height,
                                ts, frame_count, cfg.output_min, cfg.output_max);
#endif
            frame_count++;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

#ifdef HAS_OPENCV
    cv::destroyAllWindows();
#endif

    spdlog::info("Shutting down...");
    cam.stop();

    {
        std::lock_guard<std::mutex> lock(robot->data.depth_mtx);
        if (robot->data.depth_valid && !robot->data.depth_obs.empty()) {
            auto& obs = robot->data.depth_obs;
            auto [mn, mx] = std::minmax_element(obs.begin(), obs.end());
            float mean = std::accumulate(obs.begin(), obs.end(), 0.0f) / obs.size();
            spdlog::info("Final depth_obs: size={} min={:.4f} max={:.4f} mean={:.4f}",
                         obs.size(), *mn, *mx, mean);
        }
    }

    if (camera_failed) {
        spdlog::error("Done (failed).");
        return 1;
    }

    spdlog::info("Done.");
    return 0;
}

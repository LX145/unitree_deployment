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
#include <atomic>
#include <csignal>
#include <signal.h>
#include <unistd.h>
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
void on_signal(int signal_number)
{
    if (g_stop) {
        // A second signal means the SDK cleanup path is stuck. _exit is
        // async-signal-safe and lets the OS release the USB resources.
        _exit(128 + signal_number);
    }
    g_stop = 1;
}

static void install_signal_handlers()
{
    struct sigaction action {};
    action.sa_handler = on_signal;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);
}

#ifdef HAS_OPENCV
// ---------------------------------------------------------------------------
// OpenCV depth visualisation
// ---------------------------------------------------------------------------
static cv::Mat metric_depth_colormap(const cv::Mat& depth_m,
                                     float near_depth, float far_depth)
{
    cv::Mat normalized(depth_m.size(), CV_32FC1);
    for (int y = 0; y < depth_m.rows; ++y) {
        for (int x = 0; x < depth_m.cols; ++x) {
            const float depth = depth_m.at<float>(y, x);
            normalized.at<float>(y, x) = depth > 0.0f
                ? std::clamp((far_depth - depth) / (far_depth - near_depth), 0.0f, 1.0f)
                : 0.0f;
        }
    }
    cv::Mat gray;
    normalized.convertTo(gray, CV_8UC1, 255.0);
    cv::Mat color;
    cv::applyColorMap(gray, color, cv::COLORMAP_JET);
    color.setTo(cv::Scalar(0, 0, 0), depth_m <= 0.0f);
    return color;
}

static void show_source_depth_opencv(const std::vector<float>& depth_m,
                                     int w, int h,
                                     float min_depth, float max_depth)
{
    if (depth_m.size() != static_cast<std::size_t>(w * h)) return;
    cv::Mat metric(h, w, CV_32FC1, const_cast<float*>(depth_m.data()));
    cv::Mat color = metric_depth_colormap(metric, min_depth, max_depth);
    cv::putText(color, "SDK depth before policy crop/resize", cv::Point(8, 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);
    cv::imshow("RealSense Source Depth", color);
}

static void show_depth_opencv(const std::vector<float>& depth_obs,
                              int w, int h, int frame_count,
                              float output_min, float output_max,
                              float min_depth, float max_depth)
{
    const float output_range = output_max - output_min;
    cv::Mat gray(h, w, CV_8UC1);
    float center_depth = min_depth;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float value = depth_obs[y * w + x];
            const float t = std::clamp((value - output_min) / output_range, 0.0f, 1.0f);
            gray.at<uint8_t>(y, x) = static_cast<uint8_t>(std::lround(t * 255.0f));
            if (x == w / 2 && y == h / 2) {
                center_depth = min_depth + t * (max_depth - min_depth);
            }
        }
    }
    cv::Mat big;
    cv::resize(gray, big, cv::Size(), 8.0, 8.0, cv::INTER_NEAREST);
    cv::Mat display;
    cv::cvtColor(big, display, cv::COLOR_GRAY2BGR);

    char buf[128];
    snprintf(buf, sizeof(buf), "#%d %dx%d policy input: near=black far=white [%.2f, %.2f]m",
             frame_count, w, h, min_depth, max_depth);
    cv::putText(display, buf, cv::Point(4, 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(0, 0, 0), 2);
    snprintf(buf, sizeof(buf), "center=%.3fm", center_depth);
    cv::putText(display, buf, cv::Point(4, display.rows - 6),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 2);
    cv::imshow("Policy Depth Input", display);
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
    cfg.capture_debug_source_depth = true;
    spdlog::info(
        "Loaded deployment depth pipeline from {}: raw={}x{}@{}Hz, output={}x{}@{:.1f}Hz, "
        "depth=[{:.2f}, {:.2f}]m, invalid_below={:.2f}m",
        deploy_config_path.string(), cfg.raw_width, cfg.raw_height, cfg.raw_fps,
        cfg.out_width, cfg.out_height, cfg.update_hz,
        cfg.min_depth, cfg.max_depth, cfg.invalid_depth_threshold);

    auto robot = std::make_shared<isaaclab::Articulation>();

#ifdef HAS_OPENCV
    // Create the windows before starting the camera so the UI remains usable
    // while the RealSense pipeline is warming up or reconnecting.
    cv::namedWindow("Policy Depth Input", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("RealSense Source Depth", cv::WINDOW_AUTOSIZE);
    cv::Mat waiting(180, 480, CV_8UC3, cv::Scalar(32, 32, 32));
    cv::putText(waiting, "Waiting for RealSense depth...", cv::Point(35, 95),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(220, 220, 220), 1);
    cv::imshow("Policy Depth Input", waiting);
    cv::imshow("RealSense Source Depth", waiting);
    cv::waitKey(1);
#endif

    // Some HighGUI backends install their own handlers while creating windows.
    // Register ours afterwards so Ctrl+C in the launching terminal is reliable.
    install_signal_handlers();

    RealSenseDepthCamera cam(cfg, robot);
    cam.start();

    spdlog::info("Depth camera starting; waiting for first valid frame...");

    int frame_count = 0;
    bool failure_reported = false;
    while (!g_stop) {
        if (cam.has_failed()) {
            if (!failure_reported) {
                spdlog::warn(
                    "Depth stream unavailable; waiting for the background reconnect loop "
                    "(press Ctrl+C, Q, or ESC to stop)");
                failure_reported = true;
            }
        } else if (failure_reported) {
            spdlog::info("Depth stream recovered");
            failure_reported = false;
        }

        // Read latest depth from shared buffer
        std::vector<float> frame;
        std::vector<float> source_depth;
        int source_width = 0;
        int source_height = 0;
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
            if (frame_count == 0) {
                spdlog::info("First valid depth frame received. Press Ctrl+C, Q, or ESC to stop.");
            }
            show_depth_opencv(frame, cfg.out_width, cfg.out_height,
                              frame_count, cfg.output_min, cfg.output_max,
                              cfg.min_depth, cfg.max_depth);
            if (cam.get_debug_source_depth(source_depth, source_width, source_height)) {
                show_source_depth_opencv(
                    source_depth, source_width, source_height,
                    cfg.invalid_depth_threshold, cfg.max_depth);
            }
#else
            show_depth_terminal(frame, cfg.out_width, cfg.out_height,
                                ts, frame_count, cfg.output_min, cfg.output_max);
#endif
            frame_count++;
        }

#ifdef HAS_OPENCV
        // Always pump HighGUI events, including while waiting for the first
        // frame. Otherwise the windows remain blank/unresponsive.
        const int display_delay_ms = std::max(
            1, static_cast<int>(std::lround(1000.0f / cfg.update_hz)));
        const int key = cv::waitKey(display_delay_ms);
        const int key_code = key < 0 ? key : key & 0xff;
        if (key_code == 3 || key_code == 27 || key_code == 'q' || key_code == 'Q') {
            spdlog::info("Exit requested");
            g_stop = 1;
        }
#else
        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
#endif
    }

#ifdef HAS_OPENCV
    cv::destroyAllWindows();
#endif

    spdlog::info("Shutting down...");

    // librealsense can occasionally hang while destroying a pipeline after a
    // device-side hardware error. Keep the normal cleanup path, but never let
    // this standalone diagnostic utility become impossible to terminate.
    std::atomic_bool camera_stopped{false};
    std::thread shutdown_watchdog([&camera_stopped]() {
        for (int elapsed_ms = 0; elapsed_ms < 3000; elapsed_ms += 50) {
            if (camera_stopped.load()) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        spdlog::error("RealSense shutdown did not finish within 3 seconds; forcing process exit");
        _exit(0);
    });
    cam.stop();
    camera_stopped.store(true);
    shutdown_watchdog.join();

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

    spdlog::info("Done.");
    return 0;
}

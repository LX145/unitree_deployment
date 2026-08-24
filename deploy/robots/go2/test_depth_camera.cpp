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
                              int w, int h, int frame_count)
{
    // Match the visible Isaac Sim Xbox depth UI: near (0 m) is white and
    // far (2 m) is black. RealSense/OpenCV already uses a top-left origin,
    // so the tensor flip required by Isaac Sim's UI is not needed here.
    cv::Mat gray(h, w, CV_32FC1);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float v = depth_obs[y * w + x];
            gray.at<float>(y, x) = 0.5f - v;  // 1=near(0m), 0=far(2m)
        }
    }
    // near=0.0m → white (255), far=2.0m → black (0)
    gray.convertTo(gray, CV_8UC1, 255.0);

    // 4x nearest-neighbour for visibility
    cv::Mat big;
    cv::resize(gray, big, cv::Size(), 4.0, 4.0, cv::INTER_NEAREST);

    // Convert to BGR for coloured overlay text
    cv::Mat display;
    cv::cvtColor(big, display, cv::COLOR_GRAY2BGR);

    // Overlay text (white)
    char buf[128];
    snprintf(buf, sizeof(buf), "#%d  %dx%d  [white=near(0m) black=far(2m)]",
             frame_count, w, h);
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
    cv::putText(display, "0m", cv::Point(2, bar_y - 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(200, 200, 200), 1);
    cv::putText(display, "2m", cv::Point(bar_w - 20, bar_y - 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(200, 200, 200), 1);

    cv::imshow("RealSense D435i Depth", display);
}

#else
// ---------------------------------------------------------------------------
// Fallback: simple terminal output (no OpenCV)
// ---------------------------------------------------------------------------
static void show_depth_terminal(const std::vector<float>& depth_obs,
                                int w, int h,
                                double timestamp, int frame_count)
{
    std::cout << "\033[2J\033[H" << std::flush;  // clear screen
    std::cout << "=== D435i Depth  frame=" << frame_count
              << "  " << w << "x" << h
              << "  age=" << std::fixed << std::setprecision(0)
              << (now_sec() - timestamp) * 1000.0 << "ms  (Ctrl+C) ===\n";

    // Simple ASCII with half-blocks and ANSI colors
    const char* ramp = " .:-=+*#%@";
    for (int y = 0; y < h; y += 2) {
        for (int x = 0; x < w; ++x) {
            float v = depth_obs[y * w + x];
            int idx = static_cast<int>((0.5f - v) * 10.0f);
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

    RealSenseDepthCamera::Config cfg;
    cfg.enable       = true;
    cfg.monitor_only = true;
    cfg.raw_width    = 848;
    cfg.raw_height   = 480;
    cfg.raw_fps      = 60;
    cfg.out_width    = 87;
    cfg.out_height   = 58;
    cfg.history      = 1;
    cfg.update_hz    = 10.0f;
    cfg.target_fx    = 45.8394f;
    cfg.min_depth    = 0.0f;
    cfg.max_depth    = 2.0f;
    cfg.output_min   = -0.5f;
    cfg.output_max   = 0.5f;
    cfg.replace_invalid_with_max = true;
    cfg.log_distribution          = true;
    cfg.save_debug_image         = true;
    cfg.debug_save_interval_s    = 2.0f;
    cfg.debug_save_dir           = "/tmp/depth_debug";

    try {
        YAML::Node yaml = YAML::LoadFile("test_depth_cfg.yaml");
        if (yaml) cfg = RealSenseDepthCamera::Config::from_yaml(yaml);
    } catch (...) {
        spdlog::info("No test_depth_cfg.yaml, using defaults");
    }

    auto robot = std::make_shared<isaaclab::Articulation>();

    RealSenseDepthCamera cam(cfg, robot);
    cam.start();

    spdlog::info("Depth camera starting; waiting for first valid frame...");
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

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
                              frame_count);

            int key = cv::waitKey(50);
            if (key == 27 || key == 'q' || key == 'Q') {  // ESC or Q
                spdlog::info("Key pressed, exiting");
                g_stop = 1;
            }
            // NOTE: do NOT auto-detect window close via getWindowProperty —
            // it returns -1 on some systems even when the window is visible.
#else
            show_depth_terminal(frame, cfg.out_width, cfg.out_height,
                                ts, frame_count);
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

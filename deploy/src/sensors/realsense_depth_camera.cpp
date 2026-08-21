// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#include "sensors/realsense_depth_camera.h"
#include "isaaclab/assets/articulation/articulation.h"

#include <librealsense2/rs.hpp>
#include <spdlog/spdlog.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <deque>
#include <sys/stat.h>
#include <sys/types.h>

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
double now_sec()
{
    static auto start = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start).count();
}

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------
/* static */
RealSenseDepthCamera::Config RealSenseDepthCamera::Config::from_yaml(const YAML::Node& node)
{
    Config c;
    if (!node) return c;

    // control
    c.enable           = node["enable"].as<bool>(false);
    c.monitor_only     = node["monitor_only"].as<bool>(true);
    if (node["fail_behavior"])
        c.fail_behavior = node["fail_behavior"].as<std::string>();

    // raw camera
    c.raw_width  = node["raw_width"].as<int>(480);
    c.raw_height = node["raw_height"].as<int>(270);
    c.raw_fps    = node["raw_fps"].as<int>(60);

    // output
    c.out_width  = node["width"].as<int>(87);
    c.out_height = node["height"].as<int>(58);
    c.history    = node["history"].as<int>(1);
    c.update_hz  = node["update_hz"].as<float>(10.0f);

    // normalization
    c.min_depth  = node["min_depth"].as<float>(0.0f);
    c.max_depth  = node["max_depth"].as<float>(2.0f);
    c.output_min = node["output_min"].as<float>(-0.5f);
    c.output_max = node["output_max"].as<float>(0.5f);

    // processing
    c.replace_invalid_with_max = node["replace_invalid_with_max"].as<bool>(true);

    // debug
    c.save_debug_image       = node["save_debug_image"].as<bool>(false);
    c.debug_save_interval_s  = node["debug_save_interval_s"].as<float>(2.0f);
    if (node["debug_save_dir"])
        c.debug_save_dir = node["debug_save_dir"].as<std::string>();

    return c;
}

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------
RealSenseDepthCamera::RealSenseDepthCamera(const Config& cfg,
                                           std::shared_ptr<isaaclab::Articulation> robot)
    : cfg_(cfg), robot_(std::move(robot))
{
    if (!robot_) {
        throw std::runtime_error("RealSenseDepthCamera: robot pointer must not be null");
    }
}

RealSenseDepthCamera::~RealSenseDepthCamera()
{
    stop();
}

// ---------------------------------------------------------------------------
// start / stop
// ---------------------------------------------------------------------------
void RealSenseDepthCamera::start()
{
    if (running_.load()) {
        spdlog::warn("[Depth] camera already running");
        return;
    }

    ready_.store(false);
    failed_.store(false);
    running_.store(true);
    thread_ = std::thread(&RealSenseDepthCamera::capture_loop, this);
    spdlog::info("[Depth] camera thread started (update_hz={}, out={}x{}, history={})",
                 cfg_.update_hz, cfg_.out_width, cfg_.out_height, cfg_.history);
}

void RealSenseDepthCamera::stop()
{
    running_.store(false);
    ready_.store(false);
    if (thread_.joinable()) {
        thread_.join();
        spdlog::info("[Depth] camera thread stopped (processed {} frames)",
                     processed_frame_count_);
    }
}

// ---------------------------------------------------------------------------
// depth preprocessing
// ---------------------------------------------------------------------------
std::vector<float> RealSenseDepthCamera::process_depth(const uint16_t* raw,
                                                        int raw_w, int raw_h,
                                                        float depth_scale)
{
    const int w = cfg_.out_width;
    const int h = cfg_.out_height;
    std::vector<float> out(w * h);

    for (int y = 0; y < h; ++y) {
        // nearest-neighbour source pixel
        int src_y = y * raw_h / h;
        const uint16_t* row = raw + src_y * raw_w;
        for (int x = 0; x < w; ++x) {
            int src_x = x * raw_w / w;
            float d = static_cast<float>(row[src_x]) * depth_scale;

            // invalid depth → replace with max_depth
            if (d <= 0.0f && cfg_.replace_invalid_with_max) {
                d = cfg_.max_depth;
            }

            // clip
            d = std::clamp(d, cfg_.min_depth, cfg_.max_depth);

            // normalize to [output_min, output_max]
            float t = (d - cfg_.min_depth) / (cfg_.max_depth - cfg_.min_depth);
            out[y * w + x] = t * (cfg_.output_max - cfg_.output_min) + cfg_.output_min;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// debug save
// ---------------------------------------------------------------------------
void RealSenseDepthCamera::save_debug_frame(const std::vector<float>& frame)
{
    // ensure directory exists
    mkdir(cfg_.debug_save_dir.c_str(), 0755);

    static int save_index = 0;
    char fname[512];
    int idx = save_index++;

    // ---- PGM binary (P5) ----
    snprintf(fname, sizeof(fname), "%s/depth_obs_%06d.pgm",
             cfg_.debug_save_dir.c_str(), idx);
    {
        std::ofstream ofs(fname, std::ios::binary);
        if (ofs.is_open()) {
            // P5 header: P5 width height maxval
            ofs << "P5\n" << cfg_.out_width << " " << cfg_.out_height << "\n255\n";
            for (float v : frame) {
                // map [-0.5, 0.5] → [0, 255]
                float t = (v - cfg_.output_min) / (cfg_.output_max - cfg_.output_min);
                uint8_t pixel = static_cast<uint8_t>(std::clamp(t * 255.0f, 0.0f, 255.0f));
                ofs.write(reinterpret_cast<const char*>(&pixel), 1);
            }
        }
    }

    // ---- text dump ----
    snprintf(fname, sizeof(fname), "%s/depth_obs_%06d.txt",
             cfg_.debug_save_dir.c_str(), idx);
    {
        std::ofstream ofs(fname);
        if (ofs.is_open()) {
            for (size_t i = 0; i < frame.size(); ++i) {
                ofs << frame[i];
                if ((i + 1) % cfg_.out_width == 0)
                    ofs << '\n';
                else
                    ofs << ' ';
            }
        }
    }
}

// ---------------------------------------------------------------------------
// main capture loop
// ---------------------------------------------------------------------------
void RealSenseDepthCamera::capture_loop()
{
    spdlog::info("[Depth] capture_loop starting...");

    // Keep the pipeline lifetime local so its USB handles are released before
    // the capture thread exits.
    {
        // ---- start RealSense pipeline ----
        rs2::context context;
        if (context.query_devices().size() == 0) {
            spdlog::error("[Depth] no RealSense device detected");
            ready_.store(false);
            failed_.store(true);
            running_.store(false);
            return;
        }

        rs2::pipeline pipe(context);
        rs2::config rs_cfg;
        rs_cfg.enable_stream(RS2_STREAM_DEPTH,
                             cfg_.raw_width, cfg_.raw_height,
                             RS2_FORMAT_Z16, cfg_.raw_fps);

        float depth_scale = 0.001f;  // default: 1 mm
        try {
            rs2::pipeline_profile profile = pipe.start(rs_cfg);
            auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
            if (depth_sensor) {
                depth_scale = depth_sensor.get_depth_scale();
            }
            spdlog::info("[Depth] pipeline started (depth_scale={})", depth_scale);

            const auto video_profile =
                profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
            const auto intrinsics = video_profile.get_intrinsics();
            const float scale_x = static_cast<float>(cfg_.out_width) / intrinsics.width;
            const float scale_y = static_cast<float>(cfg_.out_height) / intrinsics.height;
            spdlog::info(
                "[Depth] intrinsics raw={}x{} fx={:.3f} fy={:.3f} cx={:.3f} cy={:.3f}; "
                "resized={}x{} fx={:.3f} fy={:.3f} cx={:.3f} cy={:.3f}",
                intrinsics.width, intrinsics.height,
                intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy,
                cfg_.out_width, cfg_.out_height,
                intrinsics.fx * scale_x, intrinsics.fy * scale_y,
                intrinsics.ppx * scale_x, intrinsics.ppy * scale_y);
        } catch (const rs2::error& e) {
            spdlog::error("[Depth] failed to start RealSense pipeline: {}", e.what());
            spdlog::error("[Depth] check that D435i is connected via USB 3.x");
            ready_.store(false);
            failed_.store(true);
            running_.store(false);
            return;
        }

        std::deque<std::vector<float>> history;

        // timing for rate control
        using clock = std::chrono::steady_clock;
        const auto desired_period =
            std::chrono::duration<double>(1.0 / cfg_.update_hz);
        auto next_wake = clock::now();

        while (running_.load()) {
            try {
                // ---- get frame ----
                rs2::frameset frames;
                if (!pipe.try_wait_for_frames(&frames, 1000)) {
                    spdlog::error("[Depth] no frame received for 1 second; marking camera failed");
                    {
                        std::lock_guard<std::mutex> lock(robot_->data.depth_mtx);
                        robot_->data.depth_valid = false;
                    }
                    ready_.store(false);
                    failed_.store(true);
                    running_.store(false);
                    break;
                }

                rs2::depth_frame depth = frames.get_depth_frame();
                if (!depth) {
                    spdlog::warn("[Depth] no depth frame in frameset");
                    continue;
                }

                const auto* raw = reinterpret_cast<const uint16_t*>(depth.get_data());
                int raw_w = depth.get_width();
                int raw_h = depth.get_height();

                // ---- preprocess ----
                auto frame = process_depth(raw, raw_w, raw_h, depth_scale);

                // ---- manage history ----
                history.push_back(std::move(frame));
                while (static_cast<int>(history.size()) > cfg_.history) {
                    history.pop_front();
                }

                // ---- flatten history (oldest first) ----
                std::vector<float> stacked;
                stacked.reserve(cfg_.out_width * cfg_.out_height * cfg_.history);
                for (const auto& f : history) {
                    stacked.insert(stacked.end(), f.begin(), f.end());
                }

                // ---- write to robot->data ----
                {
                    std::lock_guard<std::mutex> lock(robot_->data.depth_mtx);
                    robot_->data.depth_obs = std::move(stacked);
                    robot_->data.depth_valid = true;
                    robot_->data.depth_timestamp = now_sec();
                    robot_->data.depth_seq++;
                }
                ready_.store(true);

                processed_frame_count_++;

            } catch (const rs2::error& e) {
                spdlog::warn("[Depth] RealSense error: {}", e.what());
                {
                    std::lock_guard<std::mutex> lock(robot_->data.depth_mtx);
                    robot_->data.depth_valid = false;
                }
                ready_.store(false);
                failed_.store(true);
                running_.store(false);
                break;
            }

            // ---- optional debug save ----
            double now = now_sec();
            if (cfg_.save_debug_image && (now - last_save_time_) >= cfg_.debug_save_interval_s) {
                if (!history.empty()) {
                    save_debug_frame(history.back());
                }
                last_save_time_ = now;
            }

            // ---- rate limit ----
            next_wake += std::chrono::duration_cast<clock::duration>(desired_period);
            auto now_clock = clock::now();
            if (next_wake > now_clock) {
                std::this_thread::sleep_until(next_wake);
            } else {
                next_wake = now_clock;
            }
        }

        try {
            pipe.stop();
            spdlog::info("[Depth] pipeline stopped.");
        } catch (const rs2::error& e) {
            spdlog::warn("[Depth] pipeline stop failed: {}", e.what());
        }
    } // pipeline + device destroyed here → USB handles released
    spdlog::info("[Depth] capture_loop exited");
}

// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#include "sensors/realsense_depth_camera.h"
#include "sensors/realsense_diagnostics.h"
#include "sensors/realsense_depth_processing.h"
#include "isaaclab/assets/articulation/articulation.h"

#include <librealsense2/rs.hpp>
#include <spdlog/spdlog.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <deque>
#include <sys/stat.h>
#include <sys/types.h>

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
struct CropRect {
    int x;
    int y;
    int width;
    int height;
};

CropRect crop_for_target_intrinsics(int raw_w, int raw_h, int out_w, int out_h,
                                    float raw_fx, float raw_fy,
                                    float raw_cx, float raw_cy,
                                    float target_fx, float target_fy,
                                    float target_cx, float target_cy)
{
    CropRect crop{0, 0, raw_w, raw_h};
    if (target_fx > 0.0f) {
        crop.width = std::clamp(
            static_cast<int>(std::lround(raw_fx * out_w / target_fx)), 1, raw_w);
        const float scale_x = static_cast<float>(out_w) / crop.width;
        crop.x = std::clamp(
            static_cast<int>(std::lround(raw_cx - target_cx / scale_x)),
            0, raw_w - crop.width);
    }
    if (target_fy > 0.0f) {
        crop.height = std::clamp(
            static_cast<int>(std::lround(raw_fy * out_h / target_fy)), 1, raw_h);
        const float scale_y = static_cast<float>(out_h) / crop.height;
        crop.y = std::clamp(
            static_cast<int>(std::lround(raw_cy - target_cy / scale_y)),
            0, raw_h - crop.height);
    }
    return crop;
}

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
    c.target_fx  = node["fx"].as<float>(node["target_fx"].as<float>(0.0f));
    c.target_fy  = node["fy"].as<float>(0.0f);
    c.target_cx  = node["cx"].as<float>((c.out_width - 1.0f) * 0.5f);
    c.target_cy  = node["cy"].as<float>((c.out_height - 1.0f) * 0.5f);
    // The exported matrix describes the native ray grid. Convert its principal
    // point to policy-image coordinates after the observation-side crop.
    if (node["real_crop"] && node["render_width"] && node["render_height"] &&
        (node["render_width"].as<int>() != c.out_width ||
         node["render_height"].as<int>() != c.out_height)) {
        const auto crop = node["real_crop"];
        c.target_cx -= crop[2].as<float>();  // left
        c.target_cy -= crop[0].as<float>();  // up
    }

    // normalization
    c.min_depth  = node["min_depth"].as<float>(0.0f);
    c.max_depth  = node["max_depth"].as<float>(2.0f);
    c.output_min = node["output_min"].as<float>(-0.5f);
    c.output_max = node["output_max"].as<float>(0.5f);

    // processing
    c.filter_chain = node["filter_chain"].as<bool>(true);
    c.filter_chain_temporal = node["filter_chain_temporal"].as<bool>(false);
    c.replace_invalid_with_max = node["replace_invalid_with_max"].as<bool>(true);
    c.blur_kernel_size = node["blur_kernel_size"].as<int>(3);
    c.blur_sigma = node["blur_sigma"].as<float>(1.0f);

    // debug
    c.log_distribution       = node["log_distribution"].as<bool>(false);
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
    spdlog::info(
        "[Depth] camera thread started (update_hz={}, out={}x{}, history={}, resize=nearest, blur={}x{}, sigma={})",
        cfg_.update_hz, cfg_.out_width, cfg_.out_height, cfg_.history,
        cfg_.blur_kernel_size, cfg_.blur_kernel_size, cfg_.blur_sigma);
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
                                                        float depth_scale,
                                                        int crop_x, int crop_y,
                                                        int crop_width, int crop_height)
{
    const int w = cfg_.out_width;
    const int h = cfg_.out_height;
    std::vector<float> metric_depth(w * h);

    // Match the training/deployment contract: crop the raw image and resize
    // with nearest-neighbour sampling before applying the image blur.
    for (int y = 0; y < h; ++y) {
        const int src_y = crop_y + y * crop_height / h;
        const uint16_t* row = raw + src_y * raw_w;
        for (int x = 0; x < w; ++x) {
            const int src_x = crop_x + x * crop_width / w;
            float d = static_cast<float>(row[src_x]) * depth_scale;

            if (d <= 0.0f && cfg_.replace_invalid_with_max) {
                d = cfg_.max_depth;
            }
            metric_depth[y * w + x] = std::clamp(d, cfg_.min_depth, cfg_.max_depth);
        }
    }

    realsense_processing::gaussian_blur(
        metric_depth, w, h, cfg_.blur_kernel_size, cfg_.blur_sigma);

    std::vector<float> out(w * h);
    for (std::size_t i = 0; i < metric_depth.size(); ++i) {
        const float t = (metric_depth[i] - cfg_.min_depth) /
                        (cfg_.max_depth - cfg_.min_depth);
        out[i] = t * (cfg_.output_max - cfg_.output_min) + cfg_.output_min;
    }
    return out;
}

#ifdef ENABLE_DEPTH_STATS
static void log_depth_distribution(const std::vector<float>& depth_obs,
                                   int w, int h, float output_max,
                                   int64_t depth_seq)
{
    if (depth_obs.size() != static_cast<std::size_t>(w * h)) return;

    const float saturated_threshold = output_max - 1.0e-6f;
    const auto saturated_count = std::count_if(
        depth_obs.begin(), depth_obs.end(),
        [saturated_threshold](float value) { return value >= saturated_threshold; });
    const double saturated_percent =
        100.0 * static_cast<double>(saturated_count) / depth_obs.size();
    const double mean = std::accumulate(depth_obs.begin(), depth_obs.end(), 0.0)
                      / depth_obs.size();

    std::ostringstream row_profile;
    row_profile << std::fixed << std::setprecision(3) << '[';
    for (int y = 0; y < h; y += 2) {
        const int row_end = std::min(y + 2, h);
        double sum = 0.0;
        for (int row = y; row < row_end; ++row) {
            sum += std::accumulate(
                depth_obs.begin() + row * w,
                depth_obs.begin() + (row + 1) * w,
                0.0);
        }
        if (y != 0) row_profile << ", ";
        row_profile << sum / (static_cast<double>(row_end - y) * w);
    }
    row_profile << ']';

    spdlog::info(
        "[Depth Stats][RealSense] seq={} mean={:.3f} max_saturated={}/{} ({:.1f}%) "
        "row_pair_mean(top->bottom)={}",
        depth_seq, mean, saturated_count, depth_obs.size(), saturated_percent,
        row_profile.str());
}
#endif

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

    // Keep a supervisor alive across USB disconnects. Pipeline teardown and
    // reconnection happen only on this worker thread, never on the FSM thread.
    int consecutive_start_failures = 0;
    int total_start_failures = 0;
    int consecutive_frame_timeouts = 0;
    bool no_device_reported = false;
    auto last_start_error_log = std::chrono::steady_clock::time_point{};
    auto last_frame_timeout_log = std::chrono::steady_clock::time_point{};
    auto last_reconnect_log = std::chrono::steady_clock::time_point{};
    auto last_hardware_reset = std::chrono::steady_clock::time_point{};
    while (running_.load()) {
        // Keep each pipeline lifetime local so USB handles are released before
        // the next reconnect attempt.
        {
        // ---- start RealSense pipeline ----
        rs2::context context;
        auto devices = context.query_devices();
        if (devices.size() == 0) {
            if (!no_device_reported) {
                spdlog::error(
                    "[Depth] no RealSense device detected; retrying silently until it reappears");
                no_device_reported = true;
            }
            ready_.store(false);
            failed_.store(true);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        rs2::device device = devices.front();
        realsense_diagnostics::install_device_change_callback(context, device);

        rs2::pipeline pipe(context);
        rs2::config rs_cfg;
        rs_cfg.enable_stream(RS2_STREAM_DEPTH,
                             cfg_.raw_width, cfg_.raw_height,
                             RS2_FORMAT_Z16, cfg_.raw_fps);

        float depth_scale = 0.001f;  // default: 1 mm
        CropRect crop{0, 0, cfg_.raw_width, cfg_.raw_height};
        try {
            rs2::pipeline_profile profile = pipe.start(rs_cfg);
            auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
            if (depth_sensor) {
                depth_scale = depth_sensor.get_depth_scale();
                realsense_diagnostics::install_notification_callback(depth_sensor);
            }
            if (consecutive_frame_timeouts == 0) {
                spdlog::info("[Depth] pipeline started (depth_scale={})", depth_scale);
            }

            const auto video_profile =
                profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
            const auto intrinsics = video_profile.get_intrinsics();
            crop = crop_for_target_intrinsics(
                intrinsics.width, intrinsics.height,
                cfg_.out_width, cfg_.out_height,
                intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy,
                cfg_.target_fx, cfg_.target_fy, cfg_.target_cx, cfg_.target_cy);
            const float scale_x = static_cast<float>(cfg_.out_width) / crop.width;
            const float scale_y = static_cast<float>(cfg_.out_height) / crop.height;
            if (consecutive_frame_timeouts == 0) {
                spdlog::info(
                    "[Depth] intrinsics raw={}x{} fx={:.3f} fy={:.3f} cx={:.3f} cy={:.3f}; "
                    "crop=({}, {}) {}x{}; resized={}x{} fx={:.3f} fy={:.3f} cx={:.3f} cy={:.3f}",
                    intrinsics.width, intrinsics.height,
                    intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy,
                    crop.x, crop.y, crop.width, crop.height,
                    cfg_.out_width, cfg_.out_height,
                    intrinsics.fx * scale_x, intrinsics.fy * scale_y,
                    (intrinsics.ppx - crop.x) * scale_x,
                    (intrinsics.ppy - crop.y) * scale_y);
            }
        } catch (const rs2::error& e) {            ++consecutive_start_failures;
            ++total_start_failures;
            const auto error_now = std::chrono::steady_clock::now();
            if (last_start_error_log.time_since_epoch().count() == 0 ||
                error_now - last_start_error_log >= std::chrono::seconds(10)) {
                realsense_diagnostics::log_pipeline_error(e, total_start_failures);
                realsense_diagnostics::log_transport(device, "pipeline_start_failed");
                last_start_error_log = error_now;
            }
            ready_.store(false);
            failed_.store(true);

            // After an electrical/USB transient the device can remain
            // enumerated at 5 Gbit/s while all VIDIOC_S_FMT requests return
            // EIO. Recreating rs2::pipeline is then insufficient; periodically
            // reset the D435i firmware and let it re-enumerate.
            bool reset_requested = false;
            if (consecutive_start_failures >= 3) {
                if (last_hardware_reset.time_since_epoch().count() == 0 ||
                    error_now - last_hardware_reset >= std::chrono::seconds(10)) {
                    try {
                        spdlog::warn("[Depth] resetting D435i after repeated start failures");
                        device.hardware_reset();
                        reset_requested = true;
                        last_hardware_reset = error_now;
                    } catch (const rs2::error& reset_error) {
                        spdlog::warn("[Depth] D435i hardware reset failed: {}", reset_error.what());
                    }
                }
                consecutive_start_failures = 0;
            }
            std::this_thread::sleep_for(
                reset_requested ? std::chrono::seconds(3) : std::chrono::seconds(1));
            continue;
        }

        std::deque<std::vector<float>> history;
        bool reset_after_stop = false;

        // InstinctLab-style RealSense SDK filter chain, mirroring
        // instinct_onboard/scripts/depth_latent_publisher.py:
        //   depth -> disparity -> hole-fill -> spatial -> temporal -> depth.
        // Filtering in disparity space is what RealSense recommends; the
        // hole-filling/spatial/temporal stages inpaint gaps, denoise and
        // stabilize the stream before the policy sees it. Filters keep
        // internal state, so they must live for the whole pipeline session
        // and be applied on the capture thread.
        rs2::disparity_transform depth_to_disparity(true);
        rs2::disparity_transform disparity_to_depth(false);
        rs2::hole_filling_filter hole_filling(1);  // 1: farthest from around
        rs2::spatial_filter spatial;
        spatial.set_option(RS2_OPTION_FILTER_MAGNITUDE, 5.0f);
        spatial.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.75f);
        spatial.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 1.0f);
        spatial.set_option(RS2_OPTION_HOLES_FILL, 4.0f);
        rs2::temporal_filter temporal;
        temporal.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.6f);
        temporal.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20.0f);

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
                    ++consecutive_frame_timeouts;
                    const auto timeout_now = std::chrono::steady_clock::now();
                    if (last_frame_timeout_log.time_since_epoch().count() == 0 ||
                        timeout_now - last_frame_timeout_log >= std::chrono::seconds(10)) {
                        spdlog::error(
                            "[Depth] no frame received after pipeline start ({} consecutive attempts); "
                            "marking camera failed",
                            consecutive_frame_timeouts);
                        realsense_diagnostics::log_transport(device, "frame_timeout");
                        last_frame_timeout_log = timeout_now;
                    }
                    reset_after_stop = consecutive_frame_timeouts >= 3;
                    {
                        std::lock_guard<std::mutex> lock(robot_->data.depth_mtx);
                        robot_->data.depth_valid = false;
                    }
                    ready_.store(false);
                    failed_.store(true);
                    break;
                }

                rs2::depth_frame depth = frames.get_depth_frame();
                if (!depth) {
                    spdlog::warn("[Depth] no depth frame in frameset");
                    continue;
                }

                // Apply the InstinctLab-style filter chain (in-place on the frame).
                // Temporal smoothing is opt-in: it adds ~1 frame latency, which
                // only matches policies trained with depth delay randomization.
                if (cfg_.filter_chain) {
                    rs2::frame f = depth;
                    f = depth_to_disparity.process(f);
                    f = hole_filling.process(f);
                    f = spatial.process(f);
                    if (cfg_.filter_chain_temporal) {
                        f = temporal.process(f);
                    }
                    f = disparity_to_depth.process(f);
                    depth = f.as<rs2::depth_frame>();
                }

                const auto* raw = reinterpret_cast<const uint16_t*>(depth.get_data());
                int raw_w = depth.get_width();
                int raw_h = depth.get_height();

                // ---- preprocess ----
                auto frame = process_depth(
                    raw, raw_w, raw_h, depth_scale,
                    crop.x, crop.y, crop.width, crop.height);

#ifdef ENABLE_DEPTH_STATS
                const double stats_now = now_sec();
                if (cfg_.log_distribution &&
                    (last_distribution_log_time_ == 0.0 ||
                     stats_now - last_distribution_log_time_ >= 1.0)) {
                    log_depth_distribution(frame, cfg_.out_width, cfg_.out_height,
                                           cfg_.output_max, processed_frame_count_ + 1);
                    last_distribution_log_time_ = stats_now;
                }
#endif

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
                const bool recovered = failed_.exchange(false);
                if (recovered) {
                    spdlog::info("[Depth] RealSense stream recovered; valid frames resumed");
                }
                consecutive_start_failures = 0;
                total_start_failures = 0;
                consecutive_frame_timeouts = 0;
                no_device_reported = false;
                last_start_error_log = std::chrono::steady_clock::time_point{};
                last_frame_timeout_log = std::chrono::steady_clock::time_point{};
                last_reconnect_log = std::chrono::steady_clock::time_point{};
                last_hardware_reset = std::chrono::steady_clock::time_point{};

                processed_frame_count_++;

            } catch (const rs2::error& e) {
                spdlog::error(
                    "[Depth] RealSense streaming error: {}; function={}, args={}, exception_type={}",
                    e.what(), e.get_failed_function(), e.get_failed_args(),
                    rs2_exception_type_to_string(e.get_type()));
                realsense_diagnostics::log_transport(device, "streaming_exception");
                {
                    std::lock_guard<std::mutex> lock(robot_->data.depth_mtx);
                    robot_->data.depth_valid = false;
                }
                ready_.store(false);
                failed_.store(true);
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
            if (consecutive_frame_timeouts <= 1 || reset_after_stop) {
                spdlog::info("[Depth] pipeline stopped.");
            }
        } catch (const rs2::error& e) {
            spdlog::warn("[Depth] pipeline stop failed: {}", e.what());
        }

        if (reset_after_stop && running_.load()) {
            const auto reset_now = std::chrono::steady_clock::now();
            if (last_hardware_reset.time_since_epoch().count() == 0 ||
                reset_now - last_hardware_reset >= std::chrono::seconds(10)) {
                try {
                    spdlog::warn(
                        "[Depth] resetting D435i after {} pipelines started without producing frames",
                        consecutive_frame_timeouts);
                    device.hardware_reset();
                    last_hardware_reset = reset_now;
                    consecutive_frame_timeouts = 0;
                    std::this_thread::sleep_for(std::chrono::seconds(3));
                } catch (const rs2::error& reset_error) {
                    spdlog::warn("[Depth] D435i hardware reset failed: {}", reset_error.what());
                }
            }
        }
        } // pipeline + device destroyed here → USB handles released

        if (running_.load()) {
            const auto reconnect_now = std::chrono::steady_clock::now();
            if (last_reconnect_log.time_since_epoch().count() == 0 ||
                reconnect_now - last_reconnect_log >= std::chrono::seconds(10)) {
                spdlog::warn("[Depth] reconnecting RealSense pipeline in background");
                last_reconnect_log = reconnect_now;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    ready_.store(false);
    spdlog::info("[Depth] capture_loop exited");
}

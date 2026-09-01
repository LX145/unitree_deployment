/**
 * @file test_depth_image.cpp
 * @brief D435i depth pipeline viewer: RAW vs FILTERED vs POLICY INPUT.
 *
 * Shows two OpenCV windows side by side:
 *   - "Raw vs Filtered": the raw z16 depth (left, jet colormap) and, when the
 *     InstinctLab-style filter chain is enabled, the filtered depth (right)
 *     so the hole-fill/spatial/temporal effect is directly visible.
 *   - "Policy Input": the exact policy observation after center-crop
 *     (optional, via target_fx), nearest-neighbour resize, Gaussian blur and
 *     normalization to [output_min, output_max] — near (0 m) white, far black.
 *
 * Build:  cd deploy/robots/go2/build && cmake .. && make test_depth_image
 * Usage:  ./test_depth_image [--out 64x36] [--target-fx 0] [--max-depth 2.0]
 *                            [--no-filter] [--temporal] [--blur 3]
 */

#include "sensors/realsense_depth_processing.h"

#include <librealsense2/rs.hpp>
#include <spdlog/spdlog.h>

#include <opencv2/opencv.hpp>

#include <csignal>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

volatile sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

struct CropRect {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
};

// Same logic as realsense_depth_camera.cpp: horizontal crop so the resized
// output has the desired focal length (i.e. the training camera's FOV).
CropRect horizontal_crop_for_target_fx(int raw_w, int raw_h, int out_w,
                                       float raw_fx, float raw_cx,
                                       float target_fx)
{
    CropRect crop{0, 0, raw_w, raw_h};
    if (target_fx <= 0.0f) return crop;
    crop.width = std::clamp(static_cast<int>(std::lround(raw_fx * out_w / target_fx)), 1, raw_w);
    const float scale_x = static_cast<float>(out_w) / crop.width;
    const float target_cx = (out_w - 1.0f) * 0.5f;
    crop.x = std::clamp(static_cast<int>(std::lround(raw_cx - target_cx / scale_x)), 0, raw_w - crop.width);
    return crop;
}

// ---------------------------------------------------------------------------
// Visualisation helpers
// ---------------------------------------------------------------------------
static cv::Mat depth_to_colormap(const uint16_t* raw, int w, int h,
                                 float depth_scale, float min_d, float max_d)
{
    cv::Mat m(h, w, CV_8UC1);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float d = static_cast<float>(raw[y * w + x]) * depth_scale;
            float t = (d <= 0.0f) ? 0.0f : (d - min_d) / (max_d - min_d);
            m.at<uchar>(y, x) = static_cast<uchar>(std::clamp(t * 255.0f, 0.0f, 255.0f));
        }
    }
    cv::Mat color;
    cv::applyColorMap(m, color, cv::COLORMAP_JET);
    return color;
}

// Mirror of RealSenseDepthCamera::process_depth: crop -> nearest resize ->
// blur -> normalize to [output_min, output_max].
static std::vector<float> process_policy_input(
    const uint16_t* raw, int raw_w, int raw_h, float depth_scale,
    const CropRect& crop, int out_w, int out_h,
    float min_d, float max_d, float out_min, float out_max,
    bool replace_invalid_with_max, int blur_kernel_size, float blur_sigma)
{
    std::vector<float> metric(out_w * out_h);
    for (int y = 0; y < out_h; ++y) {
        const int src_y = y * raw_h / out_h;
        const uint16_t* row = raw + src_y * raw_w;
        for (int x = 0; x < out_w; ++x) {
            const int src_x = crop.x + x * crop.width / out_w;
            float d = static_cast<float>(row[src_x]) * depth_scale;
            if (d <= 0.0f && replace_invalid_with_max) d = max_d;
            metric[y * out_w + x] = std::clamp(d, min_d, max_d);
        }
    }

    realsense_processing::gaussian_blur(metric, out_w, out_h, blur_kernel_size, blur_sigma);

    std::vector<float> out(out_w * out_h);
    for (std::size_t i = 0; i < metric.size(); ++i) {
        const float t = (metric[i] - min_d) / (max_d - min_d);
        out[i] = t * (out_max - out_min) + out_min;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Simple argv parsing
// ---------------------------------------------------------------------------
static const char* arg_value(int& i, int argc, char** argv, const char* name)
{
    if (i + 1 >= argc) {
        spdlog::error("Missing value for {}", name);
        std::exit(1);
    }
    return argv[++i];
}

int main(int argc, char** argv)
{
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%H:%M:%S] [%^%l%$] %v");

    // ---- defaults (parkour-aligned policy input, 848x480 @ 30 Hz raw) ----
    int raw_w = 848, raw_h = 480, raw_fps = 30;
    // Default to the policy's real input: 32x36 with the training focal length
    // (matches the trained 64x36 camera cropped to its central 32 columns).
    // Use --out 64x36 --target-fx 0 for the full-frame view.
    int out_w = 32, out_h = 36;
    float target_fx = 32.27484130859375f;
    float min_d = 0.0f, max_d = 2.0f;
    float out_min = -0.5f, out_max = 0.5f;
    bool filter_chain = true;
    bool filter_chain_temporal = false;
    int blur_kernel_size = 3;
    float blur_sigma = 1.0f;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) {
            if (std::sscanf(argv[++i], "%dx%d", &out_w, &out_h) != 2) {
                spdlog::error("Bad --out (expected WxH)");
                return 1;
            }
        } else if (a == "--raw" && i + 1 < argc) {
            if (std::sscanf(argv[++i], "%dx%d", &raw_w, &raw_h) != 2) {
                spdlog::error("Bad --raw (expected WxH)");
                return 1;
            }
        } else if (a == "--fps") {
            raw_fps = std::atoi(arg_value(i, argc, argv, "--fps"));
        } else if (a == "--target-fx") {
            target_fx = std::atof(arg_value(i, argc, argv, "--target-fx"));
        } else if (a == "--max-depth") {
            max_d = std::atof(arg_value(i, argc, argv, "--max-depth"));
        } else if (a == "--no-filter") {
            filter_chain = false;
        } else if (a == "--temporal") {
            filter_chain_temporal = true;
        } else if (a == "--blur") {
            blur_kernel_size = std::atoi(arg_value(i, argc, argv, "--blur"));
        } else if (a == "--help") {
            std::printf(
                "Usage: %s [--out WxH] [--raw WxH] [--fps N] [--target-fx F]\n"
                "          [--max-depth M] [--no-filter] [--temporal] [--blur K]\n",
                argv[0]);
            return 0;
        } else {
            spdlog::warn("Ignoring unknown argument: {}", a);
        }
    }

    // ---- RealSense pipeline ----
    rs2::context ctx;
    rs2::pipeline pipe(ctx);
    rs2::config rs_cfg;
    rs_cfg.enable_stream(RS2_STREAM_DEPTH, raw_w, raw_h, RS2_FORMAT_Z16, raw_fps);

    rs2::pipeline_profile profile;
    try {
        profile = pipe.start(rs_cfg);
    } catch (const rs2::error& e) {
        spdlog::error("Pipeline start failed: {}", e.what());
        return 1;
    }
    const float depth_scale = profile.get_device().first<rs2::depth_sensor>().get_depth_scale();
    const auto intrinsics =
        profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
    const CropRect crop = horizontal_crop_for_target_fx(
        intrinsics.width, intrinsics.height, out_w, intrinsics.fx, intrinsics.ppx, target_fx);
    spdlog::info("raw={}x{} fx={:.2f} cx={:.2f} depth_scale={:.5f} crop=({}, {}) {}x{} -> out={}x{}",
                 intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.ppx, depth_scale,
                 crop.x, crop.y, crop.width, crop.height, out_w, out_h);

    // ---- InstinctLab-style filter chain (disparity space) ----
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

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    cv::namedWindow("Raw vs Filtered", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Policy Input", cv::WINDOW_AUTOSIZE);
    spdlog::info("Press Ctrl+C, Q or ESC to stop.");

    int frame_count = 0;
    while (!g_stop) {
        rs2::frameset frames;
        if (!pipe.try_wait_for_frames(&frames, 1000)) {
            continue;
        }
        rs2::depth_frame depth = frames.get_depth_frame();
        if (!depth) continue;
        const int w = depth.get_width();
        const int h = depth.get_height();

        // ---- RAW (left) ----
        cv::Mat raw_img = depth_to_colormap(
            reinterpret_cast<const uint16_t*>(depth.get_data()), w, h, depth_scale, min_d, max_d);

        // ---- FILTERED (right) / policy input source ----
        const uint16_t* source_raw = reinterpret_cast<const uint16_t*>(depth.get_data());
        cv::Mat filtered_img;
        if (filter_chain) {
            rs2::frame f = depth;
            f = depth_to_disparity.process(f);
            f = hole_filling.process(f);
            f = spatial.process(f);
            if (filter_chain_temporal) f = temporal.process(f);
            f = disparity_to_depth.process(f);
            rs2::depth_frame filtered = f.as<rs2::depth_frame>();
            source_raw = reinterpret_cast<const uint16_t*>(filtered.get_data());
            filtered_img = depth_to_colormap(source_raw, w, h, depth_scale, min_d, max_d);
        }

        // ---- window 1: raw | filtered ----
        cv::Mat combined;
        if (filter_chain) {
            cv::hconcat(raw_img, filtered_img, combined);
        } else {
            combined = raw_img;
        }
        char buf[160];
        std::snprintf(buf, sizeof(buf), "#%d  RAW  |  FILTERED (hole-fill+spatial%s)",
                      frame_count, filter_chain_temporal ? "+temporal" : "");
        cv::putText(combined, buf, cv::Point(4, 14), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);
        cv::imshow("Raw vs Filtered", combined);

        // ---- window 2: policy input (near=white, far=black) ----
        const auto obs = process_policy_input(
            source_raw, w, h, depth_scale, crop, out_w, out_h,
            min_d, max_d, out_min, out_max, true, blur_kernel_size, blur_sigma);
        cv::Mat pol(out_h, out_w, CV_32FC1);
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                pol.at<float>(y, x) = 0.5f - obs[y * out_w + x];  // near(0m)->1 -> white
            }
        }
        pol.convertTo(pol, CV_8UC1, 255.0);
        cv::Mat pol_big;
        cv::resize(pol, pol_big, cv::Size(), 4.0, 4.0, cv::INTER_NEAREST);
        cv::Mat pol_disp;
        cv::cvtColor(pol_big, pol_disp, cv::COLOR_GRAY2BGR);
        std::snprintf(buf, sizeof(buf), "#%d  POLICY INPUT %dx%d  [white=near black=far]",
                      frame_count, out_w, out_h);
        cv::putText(pol_disp, buf, cv::Point(4, 12), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(200, 200, 200), 1);
        cv::imshow("Policy Input", pol_disp);

        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') g_stop = 1;
        ++frame_count;
    }

    cv::destroyAllWindows();
    pipe.stop();
    spdlog::info("Done ({} frames shown).", frame_count);
    return 0;
}

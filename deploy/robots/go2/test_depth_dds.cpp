// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
//
// Standalone tool: subscribe to the policy-depth debug DDS topic and verify depth data.
// Usage:
//   ./test_depth_dds [--network lo] [--save] [--no-display] [--topic TOPIC]
//     --topic    DDS topic to subscribe (default: rt/depth_image_debug)
//     --save     Save depth frames as PGM files to /tmp/depth_dds/
//     --no-display  Disable OpenCV window

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/dds_wrapper/common/Subscription.h>
#include <unitree/idl/go2/HeightMap_.hpp>

#include <atomic>
#include <csignal>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <thread>
#include <opencv2/opencv.hpp>

static std::atomic<bool> g_running{true};

static void sig_handler(int) { g_running.store(false); }

int main(int argc, char** argv)
{
    signal(SIGINT, sig_handler);

    bool save_frames = false;
    bool show_display = true;  // default: show
    std::string network;
    std::string topic = "rt/depth_image_debug";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--save") save_frames = true;
        else if (arg == "--no-display") show_display = false;
        else if (arg == "--network" && i + 1 < argc) network = argv[++i];
        else if (arg == "--topic" && i + 1 < argc) topic = argv[++i];
    }

    // Initialize DDS
    unitree::robot::ChannelFactory::Instance()->Init(0, network);
    std::cout << "[test_depth_dds] DDS initialized" << std::endl;

    // Subscribe to depth topic
    int frame_count = 0;
    int width = 0, height = 0;
    float data_min = 0, data_max = 0, data_mean = 0;
    std::vector<float> latest_frame;
    std::mutex frame_mutex;

    unitree::robot::SubscriptionBase<unitree_go::msg::dds_::HeightMap_> sub(
        topic,
        [&](const void* msg) {
            auto& hm = *static_cast<const unitree_go::msg::dds_::HeightMap_*>(msg);
            std::lock_guard<std::mutex> lock(frame_mutex);
            width = hm.width();
            height = hm.height();

            if (hm.data().empty()) return;

            latest_frame = hm.data();
            frame_count++;

            // Statistics
            auto [mn, mx] = std::minmax_element(latest_frame.begin(), latest_frame.end());
            data_min = *mn;
            data_max = *mx;
            data_mean = std::accumulate(latest_frame.begin(), latest_frame.end(), 0.0f)
                      / latest_frame.size();
        });

    sub.set_timeout_ms(5000);
    std::cout << "[test_depth_dds] Waiting for " << topic << "..." << std::endl;
    sub.wait_for_connection();
    std::cout << "[test_depth_dds] Connected! Receiving depth frames..." << std::endl;

    if (show_display) {
        cv::namedWindow("DDS Depth", cv::WINDOW_NORMAL | cv::WINDOW_GUI_EXPANDED);
        cv::resizeWindow("DDS Depth", 87 * 4, 58 * 4);
        cv::Mat blank(58, 87, CV_8UC1, cv::Scalar(128));
        cv::imshow("DDS Depth", blank);
        cv::waitKey(1);
        std::cout << "[test_depth_dds] OpenCV window 'DDS Depth' opened (87x58)" << std::endl;
    }

    int last_count = 0;
    while (g_running.load()) {
        // Process at ~10 Hz to match depth update rate
        auto t0 = std::chrono::steady_clock::now();

        int snapshot_count;
        int snapshot_width;
        int snapshot_height;
        float snapshot_min;
        float snapshot_max;
        float snapshot_mean;
        std::vector<float> snapshot_frame;
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            snapshot_count = frame_count;
            snapshot_width = width;
            snapshot_height = height;
            snapshot_min = data_min;
            snapshot_max = data_max;
            snapshot_mean = data_mean;
            snapshot_frame = latest_frame;
        }

        if (snapshot_count > last_count) {
            last_count = snapshot_count;
            std::cout << "[test_depth_dds] frame=" << snapshot_count
                      << " size=" << snapshot_width << "x" << snapshot_height
                      << " min=" << snapshot_min
                      << " max=" << snapshot_max
                      << " mean=" << snapshot_mean
                      << std::endl;
        }

        // Save debug PGM
        if (save_frames && !snapshot_frame.empty()) {
            static int save_idx = 0;
            static auto last_save = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_save).count() >= 1.0) {
                last_save = now;
                char fname[256];
                snprintf(fname, sizeof(fname), "/tmp/depth_dds/depth_%04d.pgm", save_idx++);
                system("mkdir -p /tmp/depth_dds 2>/dev/null");

                std::ofstream ofs(fname, std::ios::binary);
                ofs << "P5\n" << snapshot_width << " " << snapshot_height << "\n255\n";
                for (float v : snapshot_frame) {
                    float t = (v + 0.5f);
                    uint8_t p = static_cast<uint8_t>(std::clamp(t * 255.0f, 0.0f, 255.0f));
                    ofs.write(reinterpret_cast<const char*>(&p), 1);
                }
                std::cout << "[test_depth_dds] saved " << fname << std::endl;
            }
        }

        if (show_display && snapshot_width > 0 && snapshot_height > 0) {
            if (!snapshot_frame.empty()) {
                cv::Mat img(snapshot_height, snapshot_width, CV_32FC1);
                for (int y = 0; y < snapshot_height; ++y)
                    for (int x = 0; x < snapshot_width; ++x) {
                        float v = 1.0f - (snapshot_frame[y * snapshot_width + x] + 0.5f);
                        img.at<float>(y, x) = std::clamp(v, 0.0f, 1.0f);
                    }
                cv::Mat disp;
                img.convertTo(disp, CV_8UC1, 255.0);
                cv::imshow("DDS Depth", disp);
            }
            int key = cv::waitKey(100);
            if (key == 27 || key == 'q') break;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    int final_frame_count;
    {
        std::lock_guard<std::mutex> lock(frame_mutex);
        final_frame_count = frame_count;
    }
    std::cout << "[test_depth_dds] Done. Received " << final_frame_count << " frames." << std::endl;
    return 0;
}

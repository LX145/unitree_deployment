// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
//
// Standalone tool: subscribe to rt/depth_image DDS topic and verify depth data.
// Usage:
//   ./test_depth_dds [--network lo] [--save] [--display]
//     --save     Save depth frames as PGM files to /tmp/depth_dds/
//     --display  Show depth via OpenCV window (requires OpenCV)

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/dds_wrapper/common/Subscription.h>
#include <unitree/idl/go2/HeightMap_.hpp>

#include <atomic>
#include <csignal>
#include <fstream>
#include <iostream>
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

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--save") save_frames = true;
        else if (arg == "--no-display") show_display = false;
        else if (arg == "--network" && i + 1 < argc) network = argv[++i];
    }

    // Initialize DDS
    unitree::robot::ChannelFactory::Instance()->Init(0, network);
    std::cout << "[test_depth_dds] DDS initialized" << std::endl;

    // Subscribe to depth topic
    int frame_count = 0;
    int width = 0, height = 0;
    float data_min = 0, data_max = 0, data_mean = 0;
    std::vector<float> latest_frame;

    unitree::robot::SubscriptionBase<unitree_go::msg::dds_::HeightMap_> sub(
        "rt/depth_image",
        [&](const void* msg) {
            auto& hm = *static_cast<const unitree_go::msg::dds_::HeightMap_*>(msg);
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
    std::cout << "[test_depth_dds] Waiting for rt/depth_image..." << std::endl;
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

        if (frame_count > last_count) {
            last_count = frame_count;
            std::cout << "[test_depth_dds] frame=" << frame_count
                      << " size=" << width << "x" << height
                      << " min=" << data_min
                      << " max=" << data_max
                      << " mean=" << data_mean
                      << std::endl;
        }

        // Save debug PGM
        if (save_frames && !latest_frame.empty()) {
            static int save_idx = 0;
            static auto last_save = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_save).count() >= 1.0) {
                last_save = now;
                char fname[256];
                snprintf(fname, sizeof(fname), "/tmp/depth_dds/depth_%04d.pgm", save_idx++);
                system("mkdir -p /tmp/depth_dds 2>/dev/null");

                std::ofstream ofs(fname, std::ios::binary);
                ofs << "P5\n" << width << " " << height << "\n255\n";
                for (float v : latest_frame) {
                    float t = (v + 0.5f);
                    uint8_t p = static_cast<uint8_t>(std::clamp(t * 255.0f, 0.0f, 255.0f));
                    ofs.write(reinterpret_cast<const char*>(&p), 1);
                }
                std::cout << "[test_depth_dds] saved " << fname << std::endl;
            }
        }

        if (show_display && width > 0 && height > 0) {
            if (!latest_frame.empty()) {
                cv::Mat img(height, width, CV_32FC1);
                for (int y = 0; y < height; ++y)
                    for (int x = 0; x < width; ++x) {
                        float v = 1.0f - (latest_frame[y * width + x] + 0.5f);
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

    std::cout << "[test_depth_dds] Done. Received " << frame_count << " frames." << std::endl;
    return 0;
}

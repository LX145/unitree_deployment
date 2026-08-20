// Standalone, read-only depth-policy pre-inference verifier for Go2.
//
// This process subscribes to rt/lowstate and reads a RealSense depth stream,
// then runs the same observation -> depth encoder -> GRU -> actor -> action
// post-processing path as the deployed depth policy. It never creates a
// LowCmd publisher and never writes an action to the robot.

#include "unitree_articulation.h"
#include "isaaclab/algorithms/algorithms.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "sensors/realsense_depth_camera.h"

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/dds_wrapper/robots/go2/go2_sub.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#ifndef GO2_SOURCE_DIR
#define GO2_SOURCE_DIR "."
#endif

namespace {

using Go2LowState = unitree::robot::go2::subscription::LowState;

std::atomic<bool> g_running{true};

void signal_handler(int)
{
    g_running.store(false);
}

struct Options {
    std::string network;
    std::filesystem::path policy_dir =
        std::filesystem::path(GO2_SOURCE_DIR) / "config/policy/velocity/depth_student";
    double duration_s = 0.0;
    double print_hz = 2.0;
    double depth_timeout_s = 15.0;
    float raw_action_limit = 5.0f;
};

void print_usage(const char* program)
{
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --network <iface>       DDS network interface (default: SDK auto)\n"
        << "  --policy-dir <path>     Depth policy directory\n"
        << "  --duration <seconds>    Stop automatically; 0 means run until Ctrl+C\n"
        << "  --print-hz <hz>         Console summary rate (default: 2)\n"
        << "  --depth-timeout <sec>   Wait timeout for first valid depth frame (default: 15)\n"
        << "  --raw-limit <value>     Warn if |raw action| exceeds this value (default: 5)\n"
        << "  -h, --help              Show this help\n";
}

template <typename T>
T parse_number(const std::string& text, const char* option)
{
    std::istringstream stream(text);
    T value{};
    if (!(stream >> value) || !stream.eof()) {
        throw std::runtime_error(std::string("Invalid value for ") + option + ": " + text);
    }
    return value;
}

Options parse_options(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* option) -> std::string {
            if (++i >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + option);
            }
            return argv[i];
        };

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--network") {
            options.network = require_value("--network");
        } else if (arg == "--policy-dir") {
            options.policy_dir = require_value("--policy-dir");
        } else if (arg == "--duration") {
            options.duration_s = parse_number<double>(require_value("--duration"), "--duration");
        } else if (arg == "--print-hz") {
            options.print_hz = parse_number<double>(require_value("--print-hz"), "--print-hz");
        } else if (arg == "--depth-timeout") {
            options.depth_timeout_s =
                parse_number<double>(require_value("--depth-timeout"), "--depth-timeout");
        } else if (arg == "--raw-limit") {
            options.raw_action_limit =
                parse_number<float>(require_value("--raw-limit"), "--raw-limit");
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    if (options.duration_s < 0.0 || options.print_hz <= 0.0 ||
        options.depth_timeout_s <= 0.0 || options.raw_action_limit <= 0.0f) {
        throw std::runtime_error("Numeric options must be positive (duration may be zero).");
    }
    return options;
}

struct VectorStats {
    float min = 0.0f;
    float max = 0.0f;
    float mean = 0.0f;
    float max_abs = 0.0f;
    bool finite = true;
};

VectorStats compute_stats(const std::vector<float>& values)
{
    VectorStats stats;
    if (values.empty()) {
        stats.finite = false;
        return stats;
    }

    stats.min = std::numeric_limits<float>::infinity();
    stats.max = -std::numeric_limits<float>::infinity();
    double sum = 0.0;
    for (float value : values) {
        if (!std::isfinite(value)) {
            stats.finite = false;
            continue;
        }
        stats.min = std::min(stats.min, value);
        stats.max = std::max(stats.max, value);
        stats.max_abs = std::max(stats.max_abs, std::abs(value));
        sum += value;
    }
    if (stats.finite) {
        stats.mean = static_cast<float>(sum / values.size());
    }
    return stats;
}

std::string format_vector(const std::vector<float>& values)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3) << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) stream << ", ";
        stream << values[i];
    }
    stream << ']';
    return stream.str();
}

}  // namespace

int main(int argc, char** argv)
{
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    spdlog::set_pattern("[%H:%M:%S] [%^%l%$] %v");

    try {
        const Options options = parse_options(argc, argv);
        const auto deploy_yaml = options.policy_dir / "params/deploy.yaml";
        const auto depth_onnx = options.policy_dir / "exported/policy_depth.onnx";
        const auto actor_onnx = options.policy_dir / "exported/policy_actor.onnx";
        for (const auto& required : {deploy_yaml, depth_onnx, actor_onnx}) {
            if (!std::filesystem::exists(required)) {
                throw std::runtime_error("Required file not found: " + required.string());
            }
        }

        const YAML::Node deploy_cfg = YAML::LoadFile(deploy_yaml.string());
        if (!deploy_cfg["depth_camera"]) {
            throw std::runtime_error("deploy.yaml has no depth_camera section");
        }

        spdlog::warn("READ-ONLY PRE-INFERENCE MODE: no LowCmd publisher will be created");
        spdlog::info("Policy directory: {}", options.policy_dir.string());
        spdlog::info("DDS network: {}", options.network.empty() ? "auto" : options.network);

        unitree::robot::ChannelFactory::Instance()->Init(0, options.network);
        auto lowstate = std::make_shared<Go2LowState>();
        lowstate->set_timeout_ms(1000);
        spdlog::info("Waiting for rt/lowstate...");
        lowstate->wait_for_connection();
        lowstate->update();
        spdlog::info("Connected to rt/lowstate (read-only subscriber)");

        auto robot = std::make_shared<unitree::BaseArticulation<Go2LowState::SharedPtr>>(lowstate);
        auto env = std::make_unique<isaaclab::ManagerBasedRLEnv>(deploy_cfg, robot);
        env->alg = std::make_unique<isaaclab::SplitDepthRunner>(
            depth_onnx.string(), actor_onnx.string(), robot, 5);

        auto camera_cfg = RealSenseDepthCamera::Config::from_yaml(deploy_cfg["depth_camera"]);
        camera_cfg.enable = true;
        camera_cfg.monitor_only = false;
        RealSenseDepthCamera camera(camera_cfg, robot);
        camera.start();

        const std::size_t expected_depth_size = static_cast<std::size_t>(camera_cfg.out_width) *
                                                camera_cfg.out_height * camera_cfg.history;
        spdlog::info("Waiting for first depth frame (expected {} values)...", expected_depth_size);
        const auto depth_wait_start = std::chrono::steady_clock::now();
        bool depth_ready = false;
        while (g_running.load()) {
            lowstate->update();
            robot->update();
            {
                std::lock_guard<std::mutex> lock(robot->data.depth_mtx);
                depth_ready = robot->data.depth_valid &&
                              robot->data.depth_obs.size() == expected_depth_size;
            }
            if (depth_ready) break;
            if (std::chrono::duration<double>(std::chrono::steady_clock::now() - depth_wait_start).count() >
                options.depth_timeout_s) {
                throw std::runtime_error("Timed out waiting for a valid RealSense depth frame");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (!g_running.load()) {
            camera.stop();
            return 0;
        }
        spdlog::info("Depth ready; starting full pre-inference at {:.1f} Hz", 1.0 / env->step_dt);

        lowstate->update();
        env->reset();

        using Clock = std::chrono::steady_clock;
        const auto loop_start = Clock::now();
        auto next_tick = loop_start;
        auto next_print = loop_start;
        const auto tick_period = std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(env->step_dt));
        const auto print_period = std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(1.0 / options.print_hz));

        bool previous_anomaly = false;
        std::uint64_t last_depth_seq = 0;
        auto last_depth_seen = loop_start;
        std::uint64_t step = 0;
        std::uint64_t anomaly_count = 0;
        while (g_running.load()) {
            const auto now = Clock::now();
            const double elapsed_s = std::chrono::duration<double>(now - loop_start).count();
            if (options.duration_s > 0.0 && elapsed_s >= options.duration_s) break;

            if (lowstate->isTimeout()) {
                spdlog::error("rt/lowstate timeout; pausing inference until DDS recovers");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                next_tick = Clock::now();
                continue;
            }

            lowstate->update();
            env->step();

            const std::vector<float> raw_action = env->action_manager->action();
            const std::vector<float> q_target = env->action_manager->processed_actions();
            const VectorStats raw_stats = compute_stats(raw_action);
            const bool limit_exceeded = raw_stats.max_abs > options.raw_action_limit;
            const bool anomaly = !raw_stats.finite || limit_exceeded;
            if (anomaly) anomaly_count++;

            std::vector<float> depth;
            std::uint64_t depth_seq = 0;
            double depth_age_ms = 0.0;
            {
                std::lock_guard<std::mutex> lock(robot->data.depth_mtx);
                depth = robot->data.depth_obs;
                depth_seq = robot->data.depth_seq;
            }
            if (depth_seq != last_depth_seq) {
                last_depth_seq = depth_seq;
                last_depth_seen = now;
            }
            depth_age_ms = std::chrono::duration<double, std::milli>(now - last_depth_seen).count();
            const VectorStats depth_stats = compute_stats(depth);

            if (now >= next_print || (anomaly && !previous_anomaly)) {
                if (anomaly) {
                    spdlog::warn("step={} depth_seq={} age={:.1f}ms depth=[{:.3f},{:.3f}] "
                                 "raw=[{:.3f},{:.3f}] max|a|={:.3f} anomalies={}",
                                 step, depth_seq, depth_age_ms, depth_stats.min, depth_stats.max,
                                 raw_stats.min, raw_stats.max, raw_stats.max_abs, anomaly_count);
                } else {
                    spdlog::info("step={} depth_seq={} age={:.1f}ms depth=[{:.3f},{:.3f}] "
                                 "raw=[{:.3f},{:.3f}] max|a|={:.3f} anomalies={}",
                                 step, depth_seq, depth_age_ms, depth_stats.min, depth_stats.max,
                                 raw_stats.min, raw_stats.max, raw_stats.max_abs, anomaly_count);
                }
                spdlog::info("raw_action={}", format_vector(raw_action));
                spdlog::info("q_target  ={}", format_vector(q_target));
                next_print = now + print_period;
            }

            previous_anomaly = anomaly;
            ++step;
            next_tick += tick_period;
            std::this_thread::sleep_until(next_tick);
            if (Clock::now() - next_tick > tick_period * 5) {
                next_tick = Clock::now();
            }
        }

        camera.stop();
        spdlog::info("Done: steps={}, anomalies={}", step, anomaly_count);
        return anomaly_count == 0 ? 0 : 2;
    } catch (const std::exception& error) {
        spdlog::critical("test_depth_action failed: {}", error.what());
        return 1;
    }
}

#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <unitree/dds_wrapper/common/Publisher.h>
#include <unitree/dds_wrapper/common/Subscription.h>
#include <unitree/idl/go2/HeightMap_.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#ifdef HAS_REALSENSE
#include "sensors/realsense_depth_camera.h"
#endif
#include "sensors/dds_depth_provider.h"  // always included for sim2sim

namespace {

double timing_now_sec()
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

class LowStateTimingMonitor
{
public:
    LowStateTimingMonitor()
    {
        sub_ = std::make_shared<unitree::robot::SubscriptionBase<unitree_go::msg::dds_::LowState_>>(
            "rt/lowstate",
            [this](const void* msg) {
                const auto& lowstate = *static_cast<const unitree_go::msg::dds_::LowState_*>(msg);
                tick_.store(lowstate.tick(), std::memory_order_release);
                rx_time_.store(timing_now_sec(), std::memory_order_release);
                seq_.fetch_add(1, std::memory_order_acq_rel);
            });
        spdlog::info("[TimingLog] lowstate timing monitor subscribed to rt/lowstate");
    }

    struct Snapshot {
        uint32_t tick = 0;
        double rx_time = 0.0;
        uint64_t seq = 0;
    };

    Snapshot snapshot() const
    {
        Snapshot s;
        s.tick = tick_.load(std::memory_order_acquire);
        s.rx_time = rx_time_.load(std::memory_order_acquire);
        s.seq = seq_.load(std::memory_order_acquire);
        return s;
    }

private:
    std::shared_ptr<unitree::robot::SubscriptionBase<unitree_go::msg::dds_::LowState_>> sub_;
    std::atomic<uint32_t> tick_{0};
    std::atomic<double> rx_time_{0.0};
    std::atomic<uint64_t> seq_{0};
};

LowStateTimingMonitor& lowstate_timing_monitor()
{
    static LowStateTimingMonitor monitor;
    return monitor;
}

} // namespace

namespace isaaclab {

REGISTER_OBSERVATION(gait_state)
{
    // ============================================================
    // 1. 静态变量
    // ============================================================
    static float gait_phase = 0.0f;
    static float stop_timer = 0.0f; // 依然保留缓冲，配合软着陆手感更好

    // Reset
    if (env->episode_length == 0) {
        gait_phase = 0.0f;
        stop_timer = 0.0f;
    }

    float dt = env->step_dt; 
    float freq = 2.0f;

    // ============================================================
    // 4. 指令获取与映射
    // ============================================================
    float cmd_vx = 0.f; 
    float cmd_vy = 0.f;
    float cmd_wz = 0.f;

    if (env->robot->data.joystick) {
        auto joystick = env->robot->data.joystick;
        float x_min = -1.0f, x_max = 1.0f;
        float y_min = -1.0f, y_max = 1.0f;
        float z_min = -1.0f, z_max = 1.0f;

        try {
            if (env->cfg["commands"]["base_velocity"]["ranges"]) {
                auto ranges = env->cfg["commands"]["base_velocity"]["ranges"];
                if (ranges["lin_vel_x"]) { x_min = ranges["lin_vel_x"][0].as<float>(); x_max = ranges["lin_vel_x"][1].as<float>(); }
                if (ranges["lin_vel_y"]) { y_min = ranges["lin_vel_y"][0].as<float>(); y_max = ranges["lin_vel_y"][1].as<float>(); }
                if (ranges["ang_vel_z"]) { z_min = ranges["ang_vel_z"][0].as<float>(); z_max = ranges["ang_vel_z"][1].as<float>(); }
            }
        } catch (...) {}

        auto map_axis = [](float v, float mn, float mx) {
            float vv = std::clamp(v, -1.0f, 1.0f);
            if (vv >= 0.0f) return vv * mx;
            else return -vv * mn;
        };

        cmd_vx = map_axis(joystick->ly(), x_min, x_max);
        cmd_vy = map_axis(-joystick->lx(), y_min, y_max); 
        cmd_wz = map_axis(-joystick->rx(), z_min, z_max);
    }

    // ============================================================
    // 5. 状态机与软着陆逻辑 (Soft Stop Logic)
    // ============================================================
    float cmd_vel_norm = std::sqrt(cmd_vx*cmd_vx + cmd_vy*cmd_vy);
    float cmd_ang_norm = std::abs(cmd_wz);
    
    // 判定用户意图
    bool is_cmd_moving = (cmd_vel_norm > 0.2f) || (cmd_ang_norm > 0.2f);

    // 缓冲逻辑
    if (is_cmd_moving) stop_timer = 0.5f;
    else if (stop_timer > 0.0f) stop_timer -= dt;
    bool is_in_buffer = (stop_timer > 0.0f);

    // 用户的意图是“激活步态”
    bool user_wants_active = is_cmd_moving || is_in_buffer;

    // 【核心修改】检测当前是否处于“半周期”状态
    // 如果 gait_phase > 0.05，说明腿可能在空中，即使指令停止，也必须把这一圈走完
    bool is_mid_cycle = (gait_phase > 0.05f);

    // 最终激活条件：用户想走 OR 周期没走完
    if (user_wants_active || is_mid_cycle) {
        gait_phase += dt * freq;
        
        // 处理周期循环
        if (gait_phase >= 1.0f) {
            gait_phase -= 1.0f; // 归零
            
            // 如果刚刚是靠“is_mid_cycle”强撑着走完这一圈的，
            // 现在既然已经归零（着地）了，且用户不想走，那就强制锁定在 0
            if (!user_wants_active) {
                gait_phase = 0.0f;
            }
        }
    } else {
        // 只有当 (用户不想走) 且 (不在半周期) 时，才保持静止
        gait_phase = 0.0f;
    }

    // ============================================================
    // 6. 输出
    // ============================================================
    static std::vector<float> offsets = {0.0f, 0.5f, 0.5f, 0.0f};
    std::vector<float> obs_gait;
    obs_gait.push_back(gait_phase);
    for(float offset : offsets) {
        obs_gait.push_back(std::sin(2.0f * M_PI * (gait_phase + offset)));
    }

    return obs_gait;
}

} // namespace isaaclab

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    policy_action_warmup_s_ = cfg["policy_action_warmup_s"].as<double>(0.0);
    timing_log_enabled_ = cfg["timing_log"].as<bool>(false);
    timing_log_autostart_ = cfg["timing_log_autostart"].as<bool>(true);
    timing_log_timestamp_ = cfg["timing_log_timestamp"].as<bool>(true);
    timing_log_path_ = cfg["timing_log_path"].as<std::string>(
        "../../../log/" + state_string + "_timing.csv");
    {
        std::filesystem::path timing_path(timing_log_path_);
        if (timing_path.is_relative()) {
            timing_path = param::proj_dir / timing_path;
        }
        timing_log_path_ = timing_path.lexically_normal().string();
    }
    if (timing_log_enabled_) {
        (void)lowstate_timing_monitor();
        const auto toggle_expr = cfg["timing_log_toggle"].as<std::string>("LT + start.on_pressed");
        timing_log_toggle_check_ = unitree::common::dsl::Compile(
            *unitree::common::dsl::Parser(toggle_expr).Parse());
        spdlog::info(
            "[TimingLog] {} path={} autostart={} toggle='{}'",
            state_string, timing_log_path_, timing_log_autostart_, toggle_expr);
    }

    const auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        deploy_cfg,
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    // Auto-detect split depth ONNX vs single ONNX
    auto onnx_dir = policy_dir / "exported";
    auto depth_onnx = onnx_dir / "policy_depth.onnx";
    auto actor_onnx = onnx_dir / "policy_actor.onnx";
    auto terrain_decoder_onnx = onnx_dir / "policy_terrain_decoder.onnx";

    if (std::filesystem::exists(depth_onnx) && std::filesystem::exists(actor_onnx)) {
        if (cfg["runner"].as<std::string>("") == "depth_e2e") {
            if (!std::filesystem::exists(terrain_decoder_onnx)) {
                throw std::runtime_error(
                    "E2E depth policy is missing " + terrain_decoder_onnx.string());
            }
            using TerrainMsg = unitree_go::msg::dds_::HeightMap_;
            auto terrain_publisher =
                std::make_shared<unitree::robot::PublisherBase<TerrainMsg>>("rt/terrain_decode");
            env->alg = std::make_unique<isaaclab::E2EDepthRunner>(
                depth_onnx.string(), actor_onnx.string(), terrain_decoder_onnx.string(),
                [terrain_publisher](const std::vector<float>& decoded_scan) {
                    constexpr std::size_t scan_size = 17 * 11;
                    if (decoded_scan.size() != scan_size) {
                        throw std::runtime_error(
                            "Terrain decoder produced " + std::to_string(decoded_scan.size()) +
                            " values, expected " + std::to_string(scan_size));
                    }

                    // Training target: base_z - hit_z - 0.3. Convert it into local terrain
                    // height relative to the base before publishing the HeightMap message.
                    std::vector<float> local_heights(scan_size);
                    std::transform(
                        decoded_scan.begin(), decoded_scan.end(), local_heights.begin(),
                        [](float value) { return -value - 0.3f; });

                    TerrainMsg message;
                    const auto now = std::chrono::steady_clock::now().time_since_epoch();
                    message.stamp(std::chrono::duration<double>(now).count());
                    message.frame_id("go2_base");
                    message.resolution(0.1f);
                    message.width(17);
                    message.height(11);
                    message.origin({-0.3f, -0.5f});
                    message.data(std::move(local_heights));
                    terrain_publisher->Write(message, 0);
                });
            spdlog::info(
                "[Terrain Decode] publishing 17x11 local height map on rt/terrain_decode");
        } else {
            const int depth_update_interval =
                deploy_cfg["depth_camera"]["depth_update_interval"].as<int>(1);
            env->alg = std::make_unique<isaaclab::SplitDepthRunner>(
                depth_onnx.string(), actor_onnx.string(), env->robot,
                depth_update_interval, false);
        }
    } else {
        env->alg = std::make_unique<isaaclab::OrtRunner>(onnx_dir / "policy.onnx");
    }

    // ---- depth camera/provider (runtime selection) ----
    {
        if (deploy_cfg["depth_camera"] && deploy_cfg["depth_camera"]["enable"].as<bool>(false)) {
            auto dc = deploy_cfg["depth_camera"];
#if defined(__aarch64__)
#ifdef HAS_REALSENSE
            auto cam_cfg = RealSenseDepthCamera::Config::from_yaml(dc);
            depth_provider_ = std::make_shared<RealSenseDepthCamera>(cam_cfg, env->robot);
            spdlog::info("[Depth] aarch64 detected: using RealSense provider");
            // Warm up the camera before the operator requests RL. Entry is
            // allowed only after the first processed frame is available.
            depth_provider_->start();
#else
            throw std::runtime_error(
                "Depth policy on aarch64 requires librealsense2 support at build time");
#endif
#else
            depth_provider_ = std::make_shared<DDSDepthProvider>(env->robot, dc);
            spdlog::info("[Depth] non-aarch64 detected: using DDS provider (rt/depth_image)");
            // Keep the simulated depth stream warm so its input statistics are
            // available while the robot is held in FixStand.
            depth_provider_->start();
#endif
        }
    }

    // Camera acquisition failures are fatal for the current RL state. Insert
    // this before joystick transitions so it has priority in the 1 kHz FSM loop.
    this->registered_checks.insert(
        this->registered_checks.begin(),
        std::make_pair(
            [&]()->bool{
                const bool failed = depth_provider_ && depth_provider_->has_failed();
                if (failed) {
                    spdlog::error(
                        "[RL Safety] depth provider failed in {}; requesting Passive",
                        getStateString());
                }
                return failed;
            },
            FSMStringMap.right.at("Passive")
        )
    );

    this->registered_checks.emplace_back(
        std::make_pair(
            // Keep the sideways limit tight, but allow up to ~100deg fore/aft
            // tilt for near-vertical push-off when jumping onto high platforms.
            [&]()->bool{
                constexpr float roll_limit = 1.0f;
                constexpr float pitch_limit = 1.75f;
                auto& gravity = env->robot->data.projected_gravity_b;
                const auto tilt = isaaclab::mdp::gravity_tilt_components(
                    gravity[0], gravity[1], gravity[2]);
                const bool exceeded = std::fabs(tilt.roll) > roll_limit ||
                                      std::fabs(tilt.pitch) > pitch_limit;
                if (exceeded) {
                    spdlog::warn(
                        "[RL Safety] orientation limit exceeded in {}: "
                        "roll_tilt={:.3f} rad (limit {:.3f}), "
                        "pitch_tilt={:.3f} rad (limit {:.3f}); requesting Passive",
                        getStateString(), tilt.roll, roll_limit, tilt.pitch, pitch_limit);
                }
                return exceeded;
            },
            FSMStringMap.right.at("Passive")
        )
    );
}

bool State_RLBase::can_enter()
{
#if defined(__aarch64__)
    if (depth_provider_ && !depth_provider_->is_ready()) {
        // RealSense recovery runs entirely on its background supervisor. Never
        // join or restart the camera from the 1 kHz FSM thread.
        spdlog::warn("[Depth] RL entry rejected: no valid depth frame; staying in FixStand");
        return false;
    }
#endif
    return true;
}

std::string State_RLBase::make_timing_log_path() const
{
    const std::filesystem::path base(timing_log_path_);
    if (!timing_log_timestamp_) return base.string();

    const std::filesystem::path parent = base.parent_path();
    const std::string stem = base.stem().string();
    const std::string ext = base.extension().string();

    const std::time_t now = std::time(nullptr);
    std::tm tm{};
    localtime_r(&now, &tm);
    char stamp[32] = {0};
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &tm);

    // Local wall-clock stamps have 1 s resolution; make repeated start/stop
    // toggles within the same second produce distinct files.
    std::filesystem::path candidate = parent / (stem + "_" + stamp + ext);
    for (int suffix = 1; std::filesystem::exists(candidate); ++suffix) {
        candidate = parent / (stem + "_" + stamp + "_" + std::to_string(suffix) + ext);
    }
    return candidate.string();
}

void State_RLBase::open_timing_log()
{
    if (!timing_log_enabled_) return;

    std::lock_guard<std::mutex> lock(timing_log_mtx_);
    if (timing_log_file_) return;

    std::filesystem::create_directories(
        std::filesystem::path(timing_log_path_).parent_path());

    // Each recording session gets its own timestamped file so successive runs
    // never overwrite each other.
    timing_log_active_path_ = make_timing_log_path();
    timing_log_file_ = fopen(timing_log_active_path_.c_str(), "w");
    if (!timing_log_file_) {
        timing_log_active_ = false;
        spdlog::error("[TimingLog] failed to open {}", timing_log_active_path_);
        return;
    }

    setvbuf(timing_log_file_, timing_log_buffer_, _IOFBF, sizeof(timing_log_buffer_));
    fprintf(timing_log_file_,
            "policy_step,t_policy_start,t_policy_end,policy_step_ms,"
            "depth_valid,depth_seq,depth_frame_number,depth_frame_gap,depth_source_stamp,"
            "depth_capture_time,depth_rx_time,depth_latency_ms,depth_age_ms,"
            "depth_seq_delta,depth_interval_ms,depth_wait_ms,depth_process_ms,depth_filter_ms,"
            "lowstate_tick,lowstate_monitor_tick,lowstate_rx_time,"
            "lowstate_age_ms,lowstate_tick_delta,lowstate_rx_seq\n");
    timing_log_active_ = true;
    spdlog::info("[TimingLog] recording started: {}", timing_log_active_path_);
}

void State_RLBase::close_timing_log()
{
    std::lock_guard<std::mutex> lock(timing_log_mtx_);
    if (!timing_log_file_) {
        timing_log_active_ = false;
        return;
    }
    fflush(timing_log_file_);
    fclose(timing_log_file_);
    timing_log_file_ = nullptr;
    timing_log_active_ = false;
    spdlog::info("[TimingLog] recording stopped: {}", timing_log_active_path_);
}

void State_RLBase::run()
{
    if (timing_log_enabled_ && timing_log_toggle_check_) {
        const bool toggle_pressed = timing_log_toggle_check_(lowstate->joystick);
        if (toggle_pressed && !timing_log_toggle_latched_) {
            if (timing_log_active_) {
                close_timing_log();
            } else {
                open_timing_log();
            }
            timing_log_toggle_latched_ = true;
        } else if (!toggle_pressed) {
            timing_log_toggle_latched_ = false;
        }
    }

    // Do not apply policy targets until the first depth frame arrives. Hold the
    // entry posture for this cycle; CtrlFSM will transition on provider failure.
    if ((depth_provider_ && !depth_provider_->is_ready()) ||
        !policy_action_ready_.load(std::memory_order_acquire)) {
        for (int i = 0; i < env->robot->data.joint_ids_map.size(); ++i) {
            lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = entry_joint_pos_[i];
        }
        return;
    }

    if (!rl_gains_applied_) {
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            auto& motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = env->robot->data.joint_stiffness[i];
            motor.kd() = env->robot->data.joint_damping[i];
            motor.dq() = 0;
            motor.tau() = 0;
        }
        rl_gains_applied_ = true;
        spdlog::info("[RL Warmup] policy action ready; switched to RL gains");
    }

    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}

void State_RLBase::enter()
{
    env->robot->update();
    entry_joint_pos_.assign(env->robot->data.joint_pos.data(),
                            env->robot->data.joint_pos.data() + env->robot->data.joint_pos.size());

    // During policy warmup, keep the stronger FixStand gains while holding the
    // entry posture. Switch to RL gains only when policy actions are released.
    rl_gains_applied_ = false;
    policy_action_ready_.store(false, std::memory_order_release);
    if (policy_action_warmup_s_ > 0.0) {
        const auto kp = param::config["FSM"]["FixStand"]["kp"].as<std::vector<float>>();
        const auto kd = param::config["FSM"]["FixStand"]["kd"].as<std::vector<float>>();
        const auto n = std::min(kp.size(), kd.size());
        for (std::size_t i = 0; i < n; ++i)
        {
            auto& motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = kp[i];
            motor.kd() = kd[i];
            motor.dq() = 0;
            motor.tau() = 0;
        }
    } else {
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            auto& motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = env->robot->data.joint_stiffness[i];
            motor.kd() = env->robot->data.joint_damping[i];
            motor.dq() = 0;
            motor.tau() = 0;
        }
        rl_gains_applied_ = true;
    }

    // Start depth provider (RealSense or DDS, depending on build)
    if (depth_provider_ && !depth_provider_->is_running()) {
        depth_provider_->start();
    }

    // Reset synchronously so the 1 kHz command thread can never observe actions
    // left over from the previous RL-state entry. Hold entry_joint_pos_ until the
    // first complete inference result has been published.
    env->reset();

    if (timing_log_enabled_) {
        close_timing_log();
        timing_log_toggle_latched_ = false;
        if (timing_log_autostart_) {
            open_timing_log();
        } else {
            spdlog::info(
                "[TimingLog] armed; press configured toggle to start/stop recording: {}",
                timing_log_path_);
        }
    }

    // Start policy thread
    policy_thread_running.store(true, std::memory_order_release);
    policy_thread = std::thread([this]{
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desiredDuration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);
        const auto action_release_time = clock::now() +
            std::chrono::duration_cast<clock::duration>(
                std::chrono::duration<double>(policy_action_warmup_s_));

        if (policy_action_warmup_s_ > 0.0) {
            spdlog::info(
                "[RL Warmup] holding entry posture for {:.3f}s while policy hidden state warms up",
                policy_action_warmup_s_);
        }

        uint64_t timing_step = 0;
        uint64_t prev_depth_seq = 0;
        uint32_t prev_lowstate_tick = 0;

        auto sleepTill = clock::now() + dt;
        while (policy_thread_running.load(std::memory_order_acquire))
        {
            const double t_policy_start = timing_now_sec();
            // Snapshot BEFORE the step: this is what the proprioception stream
            // looked like at the instant the policy started. Taking it after
            // env->step() (as an earlier revision did) reports packets that
            // arrived *during* the step, which makes the age meaningless.
            const auto lowstate_snapshot = lowstate_timing_monitor().snapshot();
            env->step();
            const double t_policy_end = timing_now_sec();

            {
                std::lock_guard<std::mutex> timing_lock(timing_log_mtx_);
                if (timing_log_file_) {
                bool depth_valid = false;
                uint64_t depth_seq = 0;
                uint64_t depth_frame_number = 0;
                uint64_t depth_frame_gap = 0;
                double depth_source_stamp = 0.0;
                double depth_capture_time = 0.0;
                double depth_rx_time = 0.0;
                double depth_interval_ms = 0.0;
                double depth_wait_ms = 0.0;
                double depth_process_ms = 0.0;
                double depth_filter_ms = 0.0;
                {
                    std::lock_guard<std::mutex> lock(env->robot->data.depth_mtx);
                    depth_valid = env->robot->data.depth_obs_last_read_valid;
                    depth_seq = env->robot->data.depth_obs_last_read_seq;
                    depth_frame_number = env->robot->data.depth_obs_last_read_frame_number;
                    depth_frame_gap = env->robot->data.depth_obs_last_read_frame_gap;
                    depth_source_stamp = env->robot->data.depth_obs_last_read_source_timestamp;
                    depth_capture_time = env->robot->data.depth_obs_last_read_capture_timestamp;
                    depth_rx_time = env->robot->data.depth_obs_last_read_rx_timestamp;
                    depth_interval_ms = env->robot->data.depth_obs_last_read_interval_ms;
                    depth_wait_ms = env->robot->data.depth_obs_last_read_wait_ms;
                    depth_process_ms = env->robot->data.depth_obs_last_read_process_ms;
                    depth_filter_ms = env->robot->data.depth_obs_last_read_filter_ms;
                }

                const uint32_t lowstate_tick = env->robot->data.lowstate_tick;
                const double depth_age_ms = (depth_valid && depth_rx_time > 0.0)
                    ? (t_policy_start - depth_rx_time) * 1000.0
                    : std::numeric_limits<double>::quiet_NaN();
                const double depth_latency_ms = (depth_valid && depth_rx_time > 0.0 &&
                                                 depth_capture_time > 0.0)
                    ? (depth_rx_time - depth_capture_time) * 1000.0
                    : std::numeric_limits<double>::quiet_NaN();
                const double lowstate_age_ms = (lowstate_snapshot.rx_time > 0.0)
                    ? (t_policy_start - lowstate_snapshot.rx_time) * 1000.0
                    : std::numeric_limits<double>::quiet_NaN();
                const long long depth_seq_delta = (timing_step == 0)
                    ? 0LL
                    : static_cast<long long>(depth_seq) - static_cast<long long>(prev_depth_seq);
                const long long lowstate_tick_delta = (timing_step == 0)
                    ? 0LL
                    : static_cast<long long>(lowstate_tick) - static_cast<long long>(prev_lowstate_tick);

                fprintf(timing_log_file_,
                        "%llu,%.9f,%.9f,%.3f,%d,%llu,%llu,%llu,%.9f,%.9f,%.9f,%.3f,%.3f,%lld,"
                        "%.3f,%.3f,%.3f,%.3f,"
                        "%u,%u,%.9f,%.3f,%lld,%llu\n",
                        static_cast<unsigned long long>(timing_step),
                        t_policy_start,
                        t_policy_end,
                        (t_policy_end - t_policy_start) * 1000.0,
                        depth_valid ? 1 : 0,
                        static_cast<unsigned long long>(depth_seq),
                        static_cast<unsigned long long>(depth_frame_number),
                        static_cast<unsigned long long>(depth_frame_gap),
                        depth_source_stamp,
                        depth_capture_time,
                        depth_rx_time,
                        depth_latency_ms,
                        depth_age_ms,
                        depth_seq_delta,
                        depth_interval_ms,
                        depth_wait_ms,
                        depth_process_ms,
                        depth_filter_ms,
                        lowstate_tick,
                        lowstate_snapshot.tick,
                        lowstate_snapshot.rx_time,
                        lowstate_age_ms,
                        lowstate_tick_delta,
                        static_cast<unsigned long long>(lowstate_snapshot.seq));

                prev_depth_seq = depth_seq;
                prev_lowstate_tick = lowstate_tick;
                ++timing_step;
                }
            }

            if (clock::now() >= action_release_time) {
                policy_action_ready_.store(true, std::memory_order_release);
            }
            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}

void State_RLBase::exit()
{
    policy_thread_running.store(false, std::memory_order_release);
    if (policy_thread.joinable()) {
        policy_thread.join();
    }
    policy_action_ready_.store(false, std::memory_order_release);

    // Keep the depth provider warm across state changes. In particular, never
    // block the FSM thread on RealSense pipeline teardown after a USB fault.

    close_timing_log();

    if (log_file) {
        fflush(log_file);
        fclose(log_file);
        log_file = nullptr;
    }
}

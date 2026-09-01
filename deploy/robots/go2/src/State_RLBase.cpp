#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <cmath>
#include <filesystem>
#include <stdexcept>
#ifdef HAS_REALSENSE
#include "sensors/realsense_depth_camera.h"
#endif
#include "sensors/dds_depth_provider.h"  // always included for sim2sim

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

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    // Auto-detect split depth ONNX vs single ONNX
    auto onnx_dir = policy_dir / "exported";
    auto depth_onnx = onnx_dir / "policy_depth.onnx";
    auto actor_onnx = onnx_dir / "policy_actor.onnx";

    if (std::filesystem::exists(depth_onnx) && std::filesystem::exists(actor_onnx)) {
#if defined(__aarch64__)
        constexpr bool encode_on_new_depth = false;
#else
        // MuJoCo depth is delivered asynchronously over DDS.
        constexpr bool encode_on_new_depth = true;
#endif
        env->alg = std::make_unique<isaaclab::SplitDepthRunner>(
            depth_onnx.string(), actor_onnx.string(), env->robot,
            5, encode_on_new_depth);
    } else {
        env->alg = std::make_unique<isaaclab::OrtRunner>(onnx_dir / "policy.onnx");
    }

    // ---- depth camera/provider (runtime selection) ----
    {
        auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");
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

void State_RLBase::run()
{
    // Do not apply policy targets until the first depth frame arrives. Hold the
    // entry posture for this cycle; CtrlFSM will transition on provider failure.
    if ((depth_provider_ && !depth_provider_->is_ready()) ||
        !policy_action_ready_.load(std::memory_order_acquire)) {
        for (int i = 0; i < env->robot->data.joint_ids_map.size(); ++i) {
            lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = entry_joint_pos_[i];
        }
        return;
    }

    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}

void State_RLBase::enter()
{
    // set gain
    for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
    {
        lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0;
        lowcmd->msg_.motor_cmd()[i].tau() = 0;
    }

    env->robot->update();
    entry_joint_pos_.assign(env->robot->data.joint_pos.data(),
                            env->robot->data.joint_pos.data() + env->robot->data.joint_pos.size());
    // Start depth provider (RealSense or DDS, depending on build)
    if (depth_provider_ && !depth_provider_->is_running()) {
        depth_provider_->start();
    }

    // Reset synchronously so the 1 kHz command thread can never observe actions
    // left over from the previous RL-state entry. Hold entry_joint_pos_ until the
    // first complete inference result has been published.
    policy_action_ready_.store(false, std::memory_order_release);
    env->reset();

    // Start policy thread
    policy_thread_running.store(true, std::memory_order_release);
    policy_thread = std::thread([this]{
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desiredDuration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

        auto sleepTill = clock::now() + dt;
        while (policy_thread_running.load(std::memory_order_acquire))
        {
            env->step();
            policy_action_ready_.store(true, std::memory_order_release);
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

    if (log_file) {
        fflush(log_file);
        fclose(log_file);
        log_file = nullptr;
    }
}

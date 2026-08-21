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
    
    static std::vector<float> last_gyro = {0.f, 0.f, 0.f};
    static float avg_lin_acc = 0.0f;
    static float avg_ang_acc = 0.0f;
    static int disturb_trigger_count = 0;
    static bool is_disturbed = false;

    // Reset
    if (env->episode_length == 0) {
        gait_phase = 0.0f;
        stop_timer = 0.0f;
        std::fill(last_gyro.begin(), last_gyro.end(), 0.f);
        avg_lin_acc = 0.0f;
        avg_ang_acc = 0.0f;
        disturb_trigger_count = 0;
        is_disturbed = false;
    }

    float dt = env->step_dt; 
    float freq = 2.0f;

    // ============================================================
    // 2. 获取传感器数据
    // ============================================================
    float raw_acc_x = env->robot->data.imu_acc[0];
    float raw_acc_y = env->robot->data.imu_acc[1];
    // float raw_acc_z = env->robot->data.imu_acc[2]; // Z轴通常接近9.8

    // 3.2 获取姿态四元数
    auto q = env->robot->data.root_quat_w;

    // 3.3 计算重力在机身坐标系下的分量
    // 世界坐标系重力向量 (向下)
    Eigen::Vector3f gravity_w(0.0f, 0.0f, -9.81f); 
    // 旋转到机身坐标系: g_b = R_wb^T * g_w = q.inverse * g_w
    Eigen::Vector3f gravity_b = q.conjugate() * gravity_w;

    // 3.4 计算“纯净”的运动加速度
    // 原理: 测量值 = 运动加速度 - 重力分量
    // 所以: 运动加速度 = 测量值 + 重力分量
    // 验证: 静止平放时 raw_z=+9.8, g_b_z=-9.8 -> 相加=0. 正确!
    float real_acc_x = raw_acc_x + gravity_b.x();
    float real_acc_y = raw_acc_y + gravity_b.y();

    float gyro_x = env->robot->data.root_ang_vel_b[0];
    float gyro_y = env->robot->data.root_ang_vel_b[1];
    float gyro_z = env->robot->data.root_ang_vel_b[2];

    // ============================================================
    // 3. 抗扰逻辑
    // ============================================================
    float inst_lin_acc_norm = std::sqrt(real_acc_x * real_acc_x + real_acc_y * real_acc_y);
    float inst_ang_acc_norm = std::sqrt(
        std::pow((gyro_x - last_gyro[0]) / dt, 2) +
        std::pow((gyro_y - last_gyro[1]) / dt, 2) +
        std::pow((gyro_z - last_gyro[2]) / dt, 2)
    );
    last_gyro = {gyro_x, gyro_y, gyro_z};

    float alpha = 0.2f;
    avg_lin_acc = (1.0f - alpha) * avg_lin_acc + alpha * inst_lin_acc_norm;
    avg_ang_acc = (1.0f - alpha) * avg_ang_acc + alpha * inst_ang_acc_norm;

    // static int debug_cnt = 0;
    // debug_cnt++;
    // if (debug_cnt > 20) { // 每 20 个周期 (约 0.4s) 打印一次，防止刷屏
    //     debug_cnt = 0;
    //     std::cout << "\n>>> IMU DEBUG INFO <<<" << std::endl;
    //     std::cout << "1. Raw Acc (X, Y)    : " << raw_acc_x << ", " << raw_acc_y << std::endl;
    //     std::cout << "2. Quat (W, X, Y, Z) : " << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z() << std::endl;
    //     std::cout << "3. Gravity Proj (X,Y): " << gravity_b.x() << ", " << gravity_b.y() << std::endl;
    //     std::cout << "--------------------------------" << std::endl;
    //     std::cout << "4. Real Acc (X, Y)   : " << real_acc_x << ", " << real_acc_y << " (Should be near 0)" << std::endl;
    //     std::cout << "5. Avg Filtered Norm : " << avg_lin_acc << " (Threshold: 2.0)" << std::endl;
    //     std::cout << "6. Is Disturbed?     : " << (is_disturbed ? "YES !!!" : "No") << std::endl;
    // }

    // 唤醒判定
    bool raw_wake_up = (avg_lin_acc > 2.0f) || (avg_ang_acc > 10.0f);
    if (raw_wake_up) disturb_trigger_count++;
    else disturb_trigger_count = 0;

    bool real_wake_up = (disturb_trigger_count > 10);
    bool can_sleep = (avg_lin_acc < 2.0) && (avg_ang_acc < 5.0f);

    if (real_wake_up) is_disturbed = true;
    if (can_sleep) is_disturbed = false;

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
    bool user_wants_active = is_cmd_moving || is_in_buffer || is_disturbed;

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
        env->alg = std::make_unique<isaaclab::SplitDepthRunner>(
            depth_onnx.string(), actor_onnx.string(), env->robot);
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

    // Camera acquisition failures and unreasonable raw policy actions are fatal
    // for the current RL state. Insert these before joystick transitions so the
    // safety transition has priority in the 1 kHz FSM loop.
    this->registered_checks.insert(
        this->registered_checks.begin(),
        std::make_pair(
            [&]()->bool{ return action_limit_exceeded_.load(); },
            FSMStringMap.right.at("Passive")
        )
    );
    this->registered_checks.insert(
        this->registered_checks.begin(),
        std::make_pair(
            [&]()->bool{ return depth_provider_ && depth_provider_->has_failed(); },
            FSMStringMap.right.at("Passive")
        )
    );

    this->registered_checks.emplace_back(
        std::make_pair(
            // roll_limit = 1.0 rad (~57deg, sideways is dangerous), pitch_limit = 1.5 rad (~86deg,
            // near-vertical pitch is normal when jumping onto high platforms).
            [&]()->bool{ return isaaclab::mdp::bad_orientation_roll_pitch(env.get(), 1.0f, 1.5f); },
            FSMStringMap.right.at("Passive")
        )
    );
}

bool State_RLBase::can_enter()
{
#if defined(__aarch64__)
    if (depth_provider_ && !depth_provider_->is_ready()) {
        // A previous initialization attempt may have finished with an error.
        // Reap that thread and start a fresh asynchronous attempt, but never
        // transition out of FixStand until a real frame has arrived.
        if (!depth_provider_->is_running()) {
            depth_provider_->stop();
            depth_provider_->start();
        }
        spdlog::warn("[Depth] RL entry rejected: no valid depth frame; staying in FixStand");
        return false;
    }
#endif
    return true;
}

void State_RLBase::run()
{
    // Do not apply policy targets until the first depth frame arrives, or after
    // the provider/action safety check has failed. Hold the entry posture for
    // this cycle; CtrlFSM will transition to Passive immediately on failure.
    if ((depth_provider_ && !depth_provider_->is_ready()) || action_limit_exceeded_.load()) {
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
    action_limit_exceeded_.store(false);

    // Start depth provider (RealSense or DDS, depending on build)
    if (depth_provider_ && !depth_provider_->is_running()) {
        depth_provider_->start();
    }

    // Start policy thread
    policy_thread_running = true;
    policy_thread = std::thread([this]{
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desiredDuration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

        auto sleepTill = clock::now() + dt;
        env->reset();

        while (policy_thread_running)
        {
            env->step();
            const auto raw_action = env->action_manager->action();
            for (float value : raw_action) {
                if (std::abs(value) > 10.0f) {
                    if (!action_limit_exceeded_.exchange(true)) {
                        spdlog::error("[RL Safety] raw action magnitude exceeded 10: {}", value);
                    }
                    break;
                }
            }
            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}

void State_RLBase::exit()
{
    policy_thread_running = false;
    if (policy_thread.joinable()) {
        policy_thread.join();
    }

    // Stop depth provider (if running)
    if (depth_provider_) {
#if defined(__aarch64__)
        // Keep a healthy RealSense stream warm across normal state changes.
        // A failed thread must be joined before a later entry can restart it.
        if (depth_provider_->has_failed()) {
            depth_provider_->stop();
        }
#endif
    }

    if (log_file) {
        fflush(log_file);
        fclose(log_file);
        log_file = nullptr;
    }
}

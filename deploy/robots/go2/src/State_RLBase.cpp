#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

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
    float acc_x = env->robot->data.imu_acc[0];
    float acc_y = env->robot->data.imu_acc[1];
    
    float gyro_x = env->robot->data.root_ang_vel_b[0];
    float gyro_y = env->robot->data.root_ang_vel_b[1];
    float gyro_z = env->robot->data.root_ang_vel_b[2];

    // ============================================================
    // 3. 抗扰逻辑
    // ============================================================
    float inst_lin_acc_norm = std::sqrt(acc_x * acc_x + acc_y * acc_y);
    float inst_ang_acc_norm = std::sqrt(
        std::pow((gyro_x - last_gyro[0]) / dt, 2) +
        std::pow((gyro_y - last_gyro[1]) / dt, 2) +
        std::pow((gyro_z - last_gyro[2]) / dt, 2)
    );
    last_gyro = {gyro_x, gyro_y, gyro_z};

    float alpha = 0.2f;
    avg_lin_acc = (1.0f - alpha) * avg_lin_acc + alpha * inst_lin_acc_norm;
    avg_ang_acc = (1.0f - alpha) * avg_ang_acc + alpha * inst_ang_acc_norm;

    // 唤醒判定
    bool raw_wake_up = (avg_lin_acc > 1.5f) || (avg_ang_acc > 10.0f);
    if (raw_wake_up) disturb_trigger_count++;
    else disturb_trigger_count = 0;

    bool real_wake_up = (disturb_trigger_count > 5);
    bool can_sleep = (avg_lin_acc < 0.5f) && (avg_ang_acc < 5.0f);

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
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
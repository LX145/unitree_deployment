#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        // TODO: smooth and limit the velocity commands
        cmd = key_commands[key];
    }
    return cmd;
}

}

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

    // 建议使用绝对路径，或者确认运行目录权限
    log_file = fopen("/home/zyt/unitree_rl_lab/robot_data_log.csv", "w");
    if (log_file) {
        // 设置全缓冲模式 (_IOFBF)，缓冲区大小 1MB
        setvbuf(log_file, write_buffer, _IOFBF, sizeof(write_buffer));
        static const char* joint_names[] = {
            "l_hip_pitch",   // 0
            "l_hip_roll",    // 1
            "l_hip_yaw",     // 2
            "l_knee",        // 3
            "l_ankle_pitch", // 4
            "l_ankle_roll",  // 5
            "r_hip_pitch",   // 6
            "r_hip_roll",    // 7
            "r_hip_yaw",     // 8
            "r_knee",        // 9
            "r_ankle_pitch", // 10
            "r_ankle_roll",  // 11
            "waist_yaw",         // 12
            "waist_roll",        // 13
            "waist_pitch",       // 14
            "l_shoulder_pitch",  // 15
            "l_shoulder_roll",   // 16
            "l_shoulder_yaw",    // 17
            "l_elbow",           // 18
            "l_wrist_roll",      // 19
            "l_wrist_pitch",     // 20
            "l_wrist_yaw",       // 21
            "r_shoulder_pitch",  // 22
            "r_shoulder_roll",   // 23
            "r_shoulder_yaw",    // 24
            "r_elbow",           // 25
            "r_wrist_roll",      // 26
            "r_wrist_pitch",     // 27
            "r_wrist_yaw"        // 28
        };

        // 1. 写入 CSV 表头
        fprintf(log_file, "time"); 
        
        for(size_t i = 0; i < env->robot->data.joint_ids_map.size(); i++) {
            int id = env->robot->data.joint_ids_map[i];
            const char* name;
            if (id >= 0 && id < 29) {
                name = joint_names[id];
            } else {
                name = "UNKNOWN"; // 遇到未知 ID 时的保底
            }

            // 【修改点】: 将关节名放在最前面，格式为 "关节名_数据类型"
            // 这样在 PlotJuggler 里，同一个关节的所有数据会排在一起，非常方便查看
            fprintf(log_file, ",%s_cmd,%s_state,%s_tau", name, name, name);
        }
        fprintf(log_file, "\n");
        fflush(log_file); // 强制刷新一次表头，确保文件创建
    }
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    
    // --- 记录数据 ---
    if (log_file) {
        // 2. 写入时间戳
        fprintf(log_file, "%.4f", log_step_count * 0.001);
        log_step_count++;

        for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
            int joint_idx = env->robot->data.joint_ids_map[i];
            
            // 获取数据
            double q_cmd_val = action[i]; 
            
            // 【修改点 B】: 获取实际关节位置 state_q
            double q_state_val = lowstate->msg_.motor_state()[joint_idx].q();

            // 获取力矩
            double tau_val = lowstate->msg_.motor_state()[joint_idx].tau_est();

            // 【修改点 C】: 写入数据，顺序必须与表头一致
            // 格式: ,cmd_q, state_q, tau
            fprintf(log_file, ",%.4f,%.4f,%.4f", q_cmd_val, q_state_val, tau_val);
            
            // 执行控制逻辑
            lowcmd->msg_.motor_cmd()[joint_idx].q() = q_cmd_val;
        }
        fprintf(log_file, "\n");
    } else {
        // 文件未打开时的 fallback
        for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
            lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
        }
    }
}
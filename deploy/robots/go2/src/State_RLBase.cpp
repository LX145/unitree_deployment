#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

namespace isaaclab
{

    // --------------------------------------------------------
    // 自定义 Observation: Gait (Index + Clock)
    // 对应 Python 中的: obs[45] (Index) 和 obs[46:50] (Clock)
    // --------------------------------------------------------
    REGISTER_OBSERVATION(gait_state)
    {
        // 1. 定义静态变量来维持步态相位 (Phase)
        // 假设频率 freq = 2.0Hz (根据你训练时的设定修改)
        float current_time = env->episode_length * env->step_dt;
        float freq = 2.0f;

        // 更新相位
        float phase = current_time * freq;
        phase = phase - std::floor(phase); // 取小数部分，确保在 [0, 1) 区间

        // 2. 定义步态偏移 (Trot Gait Offsets: FL, FR, RL, RR)
        // 顺序必须与训练时一致！Unitree SDK 顺序通常是 FR, FL, RR, RL 或 FL, FR, RL, RR
        // 这里假设顺序是 FL, FR, RL, RR (0, 1, 2, 3)
        // Trot: 对角线同步。FL(0) 和 RR(3) 一组，FR(1) 和 RL(2) 一组
        static std::vector<float> offsets = {0.0f, 0.5f, 0.5f, 0.0f}; 

        std::vector<float> obs_gait;
        obs_gait.reserve(5); // 1 index + 4 clock

        // Part A: Gait Index (1维) 相位
        obs_gait.push_back(phase); 

        // Part B: Gait Clock (4维)
        // Python代码: gait_clock = np.sin(2 * np.pi * phase_per_leg)
        for(float offset : offsets) {
            float p = phase + offset;
            obs_gait.push_back(std::sin(2.0f * M_PI * p));
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
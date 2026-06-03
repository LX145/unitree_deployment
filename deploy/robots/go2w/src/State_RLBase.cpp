#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#ifdef HAS_REALSENSE
#include "sensors/realsense_depth_camera.h"
#endif

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

    // ---- depth camera (monitor_only mode) ----
#ifdef HAS_REALSENSE
    {
        auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");
        if (deploy_cfg["depth_camera"] && deploy_cfg["depth_camera"]["enable"].as<bool>(false)) {
            auto cam_cfg = RealSenseDepthCamera::Config::from_yaml(deploy_cfg["depth_camera"]);
            auto cam = std::make_shared<RealSenseDepthCamera>(cam_cfg, env->robot);
            depth_camera_handle_ = cam;
            spdlog::info("[Depth] camera created (monitor_only={})", cam_cfg.monitor_only);
        }
    }
#endif

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
    for(int i(0); i < 12; i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
    for(int i(12); i < 16; i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].dq() = action[i];
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

    // Start depth camera (if configured)
#ifdef HAS_REALSENSE
    if (depth_camera_handle_) {
        static_cast<RealSenseDepthCamera*>(depth_camera_handle_.get())->start();
    }
#endif

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

    // Stop depth camera (if running)
#ifdef HAS_REALSENSE
    if (depth_camera_handle_) {
        static_cast<RealSenseDepthCamera*>(depth_camera_handle_.get())->stop();
    }
#endif

    if (log_file) {
        fflush(log_file);
        fclose(log_file);
        log_file = nullptr;
    }
}

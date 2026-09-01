// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <spdlog/spdlog.h>
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/manager/action_manager.h"

namespace isaaclab
{

class JointAction : public ActionTerm
{
public:
    JointAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :ActionTerm(cfg, env)
    {
        if(cfg["joint_ids"].IsNull()) {
            _action_dim = env->robot->data.joint_ids_map.size();
        } else {
            _joint_ids = cfg["joint_ids"].as<std::vector<int>>();
            _action_dim = _joint_ids.size();
        }
        _raw_actions.resize(_action_dim, 0.0f);
        _processed_actions.resize(_action_dim, 0.0f);
        if(!cfg["scale"].IsNull()) {
            _scale = cfg["scale"].as<std::vector<float>>();
        }
        if(!cfg["offset"].IsNull()) {
            _offset = cfg["offset"].as<std::vector<float>>();
        }
        if(!cfg["clip"].IsNull()) {
            _clip = cfg["clip"].as<std::vector<std::vector<float> >>();
        }
    }

    virtual void process_actions(std::vector<float> actions)
    {
        // TODO: modify action by joint_ids
        _raw_actions = actions;
        int clipped_count = 0;
        int first_clipped_index = -1;
        float first_value = 0.0f;
        float first_lower = 0.0f;
        float first_upper = 0.0f;
        for(int i(0); i<_action_dim; ++i)
        {
            float action = _raw_actions[i];
            if(!_clip.empty()) {
                const float lower = _clip[i][0];
                const float upper = _clip[i][1];
                if (action < lower || action > upper) {
                    if (first_clipped_index < 0) {
                        first_clipped_index = i;
                        first_value = action;
                        first_lower = lower;
                        first_upper = upper;
                    }
                    ++clipped_count;
                }
                action = std::clamp(action, lower, upper);
            }
            if(!_scale.empty()) {
                _processed_actions[i] = action * _scale[i];
            } else {
                _processed_actions[i] = action;
            }
            if(!_offset.empty()) {
                _processed_actions[i] += _offset[i];
            }
        }
        if(!_clip.empty())
        {
            // Action processing runs in the policy thread. Keep this warning
            // rate-limited so an abnormal policy output cannot flood logs.
            if (clipped_count > 0) {
                static auto last_clip_warning = std::chrono::steady_clock::time_point{};
                const auto now = std::chrono::steady_clock::now();
                if (last_clip_warning.time_since_epoch().count() == 0 ||
                    now - last_clip_warning >= std::chrono::seconds(1)) {
                    spdlog::warn(
                        "[Action] clip triggered: {} action(s), first index={} "
                        "value={:.4f}, limit=[{:.4f}, {:.4f}]",
                        clipped_count, first_clipped_index, first_value,
                        first_lower, first_upper);
                    last_clip_warning = now;
                }
            }
        }
    }


    int action_dim() 
    {
        return _action_dim;
    }

    std::vector<float> raw_actions() 
    {
        return _raw_actions;
    }
    
    std::vector<float> processed_actions() 
    {
        return _processed_actions;
    }

    void reset()
    {
        _raw_actions.assign(_action_dim, 0.0f);
        _processed_actions.assign(_action_dim, 0.0f);
        if(!_offset.empty()) {
            _processed_actions = _offset;
        }
    }

protected:
    int _action_dim;
    std::vector<int> _joint_ids;

    std::vector<float> _raw_actions;
    std::vector<float> _processed_actions;

    std::vector<float> _scale;
    std::vector<float> _offset;
    std::vector<std::vector<float> > _clip;
};


class JointPositionAction : public JointAction
{
public:
    JointPositionAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :JointAction(cfg, env)
    {
    }
};

class JointVelocityAction : public JointAction
{
public:
    JointVelocityAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :JointAction(cfg, env)
    {
    }
};

REGISTER_ACTION(JointPositionAction);
REGISTER_ACTION(JointVelocityAction);

};

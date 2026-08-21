#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

inline bool bad_orientation(ManagerBasedRLEnv* env, float limit_angle = 1.0)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::fabs(std::acos(-data[2])) > limit_angle;
}

// Roll/pitch separated base-orientation check.
// g_b = (gx, gy, gz) is gravity in the base frame (unit vector, pointing down
// when upright, i.e. g_b = (0, 0, -1)).
//   roll  = atan2(-gy, -gz)  // tilt about the x axis (sideways), 0 = upright
//   pitch = atan2( gx, -gz)  // tilt about the y axis (fore/aft),   0 = upright
// Allows a large pitch (e.g. near-vertical push-off when jumping onto high
// platforms) while keeping the roll threshold tight.
inline bool bad_orientation_roll_pitch(ManagerBasedRLEnv* env, float roll_limit, float pitch_limit)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    float gx = data[0], gy = data[1], gz = data[2];
    float roll  = std::atan2(-gy, -gz);
    float pitch = std::atan2( gx, -gz);
    return std::fabs(roll) > roll_limit || std::fabs(pitch) > pitch_limit;
}

} 
} 
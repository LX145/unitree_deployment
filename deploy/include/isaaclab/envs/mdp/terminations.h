#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

#include <algorithm>
#include <cmath>

namespace isaaclab
{
namespace mdp
{

inline bool bad_orientation(ManagerBasedRLEnv* env, float limit_angle = 1.0)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    const float cosine = std::clamp(-data[2], -1.0f, 1.0f);
    return std::acos(cosine) > limit_angle;
}

struct TiltComponents
{
    float roll;
    float pitch;
};

// Decompose the total tilt represented by gravity into a tilt vector. Unlike
// two independent atan2 projections, this remains continuous when gz crosses
// zero (the base passes through a vertical attitude).
inline TiltComponents gravity_tilt_components(float gx, float gy, float gz)
{
    const float tilt = std::acos(std::clamp(-gz, -1.0f, 1.0f));
    const float horizontal = std::hypot(gx, gy);

    if (horizontal > 1.0e-6f) {
        return {tilt * (-gy) / horizontal, tilt * gx / horizontal};
    }

    if (tilt < 1.0e-6f) {
        return {0.0f, 0.0f};
    }

    // At exactly upside-down the tilt direction is undefined. Returning the
    // full tilt on both axes ensures that this singular attitude fails any
    // meaningful roll/pitch safety limits instead of appearing upright.
    return {tilt, tilt};
}

// Roll/pitch-separated base-orientation check. This permits a larger fore/aft
// tilt for jumping while retaining a tighter sideways safety limit.
inline bool bad_orientation_roll_pitch(ManagerBasedRLEnv* env, float roll_limit, float pitch_limit)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    const auto components = gravity_tilt_components(data[0], data[1], data[2]);
    return std::fabs(components.roll) > roll_limit ||
           std::fabs(components.pitch) > pitch_limit;
}

} 
} 
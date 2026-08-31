// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <vector>

namespace realsense_processing {

/// Apply a channel-free Gaussian blur with replicated image borders.
void gaussian_blur(std::vector<float>& image,
                   int width,
                   int height,
                   int kernel_size,
                   float sigma);

} // namespace realsense_processing

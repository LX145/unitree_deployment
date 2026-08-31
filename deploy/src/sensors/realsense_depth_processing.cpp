// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#include "sensors/realsense_depth_processing.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace realsense_processing {

void gaussian_blur(std::vector<float>& image,
                   int width,
                   int height,
                   int kernel_size,
                   float sigma)
{
    if (kernel_size <= 1) {
        return;
    }
    if (kernel_size % 2 == 0 || sigma <= 0.0f) {
        throw std::invalid_argument(
            "RealSense Gaussian blur requires a positive sigma and odd kernel size");
    }
    if (width <= 0 || height <= 0 ||
        image.size() != static_cast<std::size_t>(width * height)) {
        throw std::invalid_argument("RealSense Gaussian blur image dimensions are invalid");
    }

    const int radius = kernel_size / 2;
    std::vector<float> kernel(kernel_size);
    float kernel_sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        const float value = std::exp(-(static_cast<float>(i * i)) /
                                     (2.0f * sigma * sigma));
        kernel[i + radius] = value;
        kernel_sum += value;
    }
    for (float& value : kernel) {
        value /= kernel_sum;
    }

    std::vector<float> horizontal(image.size());
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float value = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                const int sample_x = std::clamp(x + k, 0, width - 1);
                value += image[y * width + sample_x] * kernel[k + radius];
            }
            horizontal[y * width + x] = value;
        }
    }

    std::vector<float> blurred(image.size());
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float value = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                const int sample_y = std::clamp(y + k, 0, height - 1);
                value += horizontal[sample_y * width + x] * kernel[k + radius];
            }
            blurred[y * width + x] = value;
        }
    }
    image.swap(blurred);
}

} // namespace realsense_processing

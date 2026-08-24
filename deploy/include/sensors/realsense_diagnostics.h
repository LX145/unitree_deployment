// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <librealsense2/rs.hpp>

namespace realsense_diagnostics {

/// Log the currently visible device/USB transport information.
void log_transport(const rs2::device& device, const char* reason);

/// Log detailed librealsense call-site information for a pipeline failure.
void log_pipeline_error(const rs2::error& error, int total_failures);

/// Install a rate-limited callback for firmware, hardware, and UVC notifications.
void install_notification_callback(const rs2::depth_sensor& sensor);

} // namespace realsense_diagnostics

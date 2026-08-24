// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#include "sensors/realsense_diagnostics.h"

#include <spdlog/spdlog.h>

#include <chrono>
#include <mutex>
#include <string>

namespace realsense_diagnostics {

void log_transport(const rs2::device& device, const char* reason)
{
    try {
        const auto info = [&device](rs2_camera_info field) -> std::string {
            return device.supports(field) ? device.get_info(field) : "unavailable";
        };
        spdlog::error(
            "[Depth] disconnect diagnostics: reason={}, name={}, serial={}, firmware={}, "
            "usb_type={}, physical_port={}",
            reason,
            info(RS2_CAMERA_INFO_NAME),
            info(RS2_CAMERA_INFO_SERIAL_NUMBER),
            info(RS2_CAMERA_INFO_FIRMWARE_VERSION),
            info(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR),
            info(RS2_CAMERA_INFO_PHYSICAL_PORT));
    } catch (const rs2::error& error) {
        spdlog::error(
            "[Depth] disconnect diagnostics: reason={}, device information unavailable ({})",
            reason, error.what());
    }
}

void log_pipeline_error(const rs2::error& error, int total_failures)
{
    spdlog::error(
        "[Depth] failed to start RealSense pipeline ({} failures): {}; "
        "function={}, args={}, exception_type={}; retries continue in background",
        total_failures,
        error.what(),
        error.get_failed_function(),
        error.get_failed_args(),
        rs2_exception_type_to_string(error.get_type()));
}

void install_notification_callback(const rs2::depth_sensor& sensor)
{
    sensor.set_notifications_callback([](rs2::notification notification) {
        static std::mutex log_mutex;
        static auto last_log = std::chrono::steady_clock::time_point{};

        std::lock_guard<std::mutex> lock(log_mutex);
        const auto now = std::chrono::steady_clock::now();
        if (last_log.time_since_epoch().count() != 0 &&
            now - last_log < std::chrono::seconds(5)) {
            return;
        }
        last_log = now;

        spdlog::warn(
            "[Depth] D435i notification: category={}, severity={}, description={}, data={}",
            rs2_notification_category_to_string(notification.get_category()),
            rs2_log_severity_to_string(notification.get_severity()),
            notification.get_description(),
            notification.get_serialized_data());
    });
}

} // namespace realsense_diagnostics

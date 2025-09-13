#pragma once

#include <format>

namespace ty {
    enum class DeviceType { CPU, CUDA };

    struct Device {
        DeviceType type;
        int index;

        bool operator!=(Device device) const {
            return type != device.type || index != device.index;
        }
        bool operator==(Device device) const {
            return type == device.type && index == device.index;
        }
    };

    std::string deviceToString(Device device);
}; // namespace ty
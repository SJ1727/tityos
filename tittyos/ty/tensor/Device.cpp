#include "tittyos/ty/tensor/Device.h"

namespace ty {
    std::string deviceToString(Device device) {
        std::string deviceType = "";

        switch (device.type()) {
        case DeviceType::CPU:
            deviceType = "cpu";
            break;
        case DeviceType::CUDA:
            deviceType = "cuda";
            break;
        default:
            break;
        }

        // With index
        if (device.index() != -1) {
            return std::format("{}:{}", deviceType, device.index());
        }

        // No index
        return std::format("{}", deviceType);
    }
} // namespace ty
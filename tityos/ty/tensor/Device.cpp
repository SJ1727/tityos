#include "tityos/ty/tensor/Device.h"

namespace ty {
    std::string deviceToString(Device device) {
        std::string deviceType = "";

        switch (device.type) {
        case DeviceType::CPU:
            deviceType = "cpu";
            break;
        case DeviceType::CUDA:
            deviceType = "cuda";
            break;
        default:
            break;
        }

        return std::format("{}:{}", deviceType, device.index);
    }
}
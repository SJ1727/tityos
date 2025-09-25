#pragma once

#include <cstdint>
#include <format>

#include "tittyos/ty/common/defines.h"

namespace ty {
    enum class tittyos_API DeviceType { CPU, CUDA };

    class tittyos_API Device {
      private:
        DeviceType type_;
        int8_t index_;

      public:
        Device(DeviceType type, int8_t index = -1) : type_(type), index_(index) {}

        bool operator!=(Device other) const noexcept {
            return this->type_ != other.type_ || this->index_ != other.index_;
        }
        bool operator==(Device other) const noexcept {
            return this->type_ == other.type_ && this->index_ == other.index_;
        }


        DeviceType type() const noexcept {
            return type_;
        }

        int8_t index() const noexcept {
            return index_;
        }

        bool isCpu() const noexcept {
            return type_ == DeviceType::CPU;
        }

        bool isCuda() const noexcept {
            return type_ == DeviceType::CUDA;
        }
    };

    std::string deviceToString(Device device);
}; // namespace ty
#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>
#include <stdexcept>

#include "tittyos/ty/tensor/DType.h"
#include "tittyos/ty/tensor/Device.h"

namespace ty {
    class Storage {
      private:
        void *dataPtr_;
        size_t numBytes_;
        DType dtype_;
        Device device_;

      public:
        Storage(size_t numBytes, DType dtype, Device device = {DeviceType::CPU, 0});

        Storage(const Storage &storage) : Storage(storage.numBytes_, storage.dtype_) {}
        Storage(Storage &&storage) noexcept;

        void operator=(const Storage &storage);
        void operator=(Storage &&storage) noexcept;

        ~Storage() noexcept;

        void *get(size_t idx) {
            return reinterpret_cast<std::byte *>(dataPtr_) + idx * dtypeSize(dtype_);
        }
        const void *get(size_t idx) const {
            return reinterpret_cast<std::byte *>(dataPtr_) + idx * dtypeSize(dtype_);
        }

        DType dtype() const {
            return dtype_;
        }

        Device device() const {
            return device_;
        }

      private:
        void allocate();
        void deallocate();
    };
}; // namespace ty
#include "tityos/ty/tensor/backend/Storage.h"

namespace ty {
    Storage::Storage(size_t numBytes, DType dtype, Device device)
        : numBytes_(numBytes), dtype_(dtype), device_(device) {
        allocate();
    }

    Storage::Storage(Storage &&storage) noexcept
        : numBytes_(storage.numBytes_), dtype_(storage.dtype_), device_(storage.device_) {
        dataPtr_ = storage.dataPtr_;
        storage.dataPtr_ = nullptr;
        storage.numBytes_ = 0;
    }

    Storage::~Storage() {
        deallocate();
    }

    void Storage::operator=(const Storage &storage) {
        numBytes_ = storage.numBytes_;
        dtype_ = storage.dtype_;

        allocate();
    }

    void Storage::operator=(Storage &&storage) noexcept {
        numBytes_ = storage.numBytes_;
        dtype_ = storage.dtype_;
        dataPtr_ = storage.dataPtr_;
    }

    void Storage::allocate() {
        if (device_.type == ty::DeviceType::CPU) {
            dataPtr_ = malloc(numBytes_);

            if (dataPtr_ == NULL) {
                throw std::bad_alloc();
            }
        } else if (device_.type == ty::DeviceType::CUDA) {
            /* Not Implemented */
        }
    }

    void Storage::deallocate() {
        if (device_.type == ty::DeviceType::CPU) {
            if (dataPtr_ != nullptr) {
                free(dataPtr_);
            }
        } else if (device_.type == ty::DeviceType::CUDA) {
            /* Not Implemented */
        } 
    }
}; // namespace ty
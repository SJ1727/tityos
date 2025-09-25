#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <stack>
#include <vector>

#include "tityos/ty/common/defines.h"
#include "tityos/ty/tensor/DType.h"
#include "tityos/ty/tensor/Device.h"
#include "tityos/ty/tensor/Shape.h"
#include "tityos/ty/tensor/backend/Storage.h"

namespace ty {
    class TITYOS_API Tensor {
        using BackwardFunc =
            std::function<void(std::vector<std::shared_ptr<Tensor>> &context, Tensor *grad)>;

      private:
        std::shared_ptr<Storage> dataStorage_;
        Shape shape_;
        std::array<int64_t, tensorMaxDims> strides_;
        int64_t offset_;

        /*--- Autograd ---*/
        bool requiresGrad_;
        BackwardFunc backwardFunc_ = nullptr;
        std::vector<std::shared_ptr<Tensor>> contextTensors_;
        std::shared_ptr<Tensor> grad_ = nullptr;

      public:
        Tensor() = default; /* Change this later */

        Tensor(Shape shape, DType dtype = ty::DType::float32, Device device = {DeviceType::CPU, 0},
               bool requiresGrad = false);

        Tensor(const Tensor &tensor) = default;
        Tensor(Tensor &&tensor) noexcept = default;

        template <typename Type>
        Tensor(Type *data, Shape shape, DType dtype = ty::DType::float32,
               Device device = {DeviceType::CPU, 0}, bool requiresGrad = false)
            : shape_(shape), offset_(0), requiresGrad_(requiresGrad) {
            if (sizeof(Type) != dtypeSize(dtype)) {
                throw std::runtime_error(
                    "Datatype size of data does not match size of tensor datatype");
            }

            calculateStrides();

            // Copying the data
            dataStorage_ =
                std::make_shared<Storage>(shape_.numElements() * dtypeSize(dtype), dtype, device);
            std::memcpy(dataStorage_->get(0), data, shape_.numElements() * dtypeSize(dtype));

            if (requiresGrad_) {
                this->zeroGrad();
            }
        }

        template <typename Type>
        Tensor(const std::vector<Type> &data, Shape shape, DType dtype = ty::DType::float32,
               Device device = {DeviceType::CPU, 0}, bool requiresGrad = false)
            : Tensor(data.data(), shape, dtype, device, requiresGrad) {}

        Tensor &operator=(const Tensor &other) = default;
        Tensor &operator=(Tensor &&other) noexcept = default;

        ~Tensor() = default;

        void *get(size_t idx) {
            return dataStorage_->get(idx);
        }

        const void *get(size_t idx) const {
            return dataStorage_->get(idx);
        }

        Shape shape() const {
            return shape_;
        }

        size_t numDims() const {
            return shape_.numDims();
        }

        int64_t numElements() const {
            return shape_.numElements();
        }

        const std::array<int64_t, tensorMaxDims> &strides() const {
            return strides_;
        }

        int64_t offset() const {
            return offset_;
        }

        DType dtype() const {
            return dataStorage_->dtype();
        }

        Device device() const {
            return dataStorage_->device();
        }

        /*--- Autograd ---*/
        std::shared_ptr<Tensor> grad() {
            return grad_;
        }

        bool requiresGrad() const {
            return requiresGrad_;
        }

        void setBackwardFunc(BackwardFunc func) {
            backwardFunc_ = func;
        }

        void setContextTensors(std::vector<std::shared_ptr<Tensor>> context) {
            contextTensors_ = context;
        }

        void zeroGrad() {
            std::vector<float> zero(this->numElements(), 0.0f);

            grad_ = std::make_shared<Tensor>(zero, shape_, this->dtype());
        }

        void setRequiresGrad(bool requiresGrad);

        void backward(std::shared_ptr<Tensor> initialGrad);

      private:
        void calculateStrides();
    };
} // namespace ty
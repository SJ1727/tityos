#include "tityos/ty/ops/BinaryOps.h"

namespace ty {
    Tensor add(Tensor &tensor1, Tensor &tensor2) {
        assertSameDevice(tensor1, tensor2);
        assertBroadcastable(tensor1.shape(), tensor2.shape());

        switch (tensor1.device().type()) {
        case DeviceType::CPU:
            return cpuAdd(tensor1, tensor2);
        case DeviceType::CUDA:
            return cudaAdd(tensor1, tensor2);
        default:
            break;
        }

        return std::move(tensor1);
    }

    Tensor subtract(Tensor &tensor1, Tensor &tensor2) {
        assertSameDevice(tensor1, tensor2);
        assertBroadcastable(tensor1.shape(), tensor2.shape());

        switch (tensor1.device().type()) {
        case DeviceType::CPU:
            return cpuSubtract(tensor1, tensor2);
        case DeviceType::CUDA:
            return cudaSubtract(tensor1, tensor2);
        default:
            break;
        }

        return std::move(tensor1);
    }

    Tensor multiply(Tensor &tensor1, Tensor &tensor2) {
        assertSameDevice(tensor1, tensor2);
        assertBroadcastable(tensor1.shape(), tensor2.shape());

        switch (tensor1.device().type()) {
        case DeviceType::CPU:
            return cpuMultiply(tensor1, tensor2);
        case DeviceType::CUDA:
            return cudaMultiply(tensor1, tensor2);
        default:
            break;
        }

        return std::move(tensor1);
    }

    Tensor divide(Tensor &tensor1, Tensor &tensor2) {
        assertSameDevice(tensor1, tensor2);
        assertBroadcastable(tensor1.shape(), tensor2.shape());

        switch (tensor1.device().type()) {
        case DeviceType::CPU:
            return cpuDivide(tensor1, tensor2);
        case DeviceType::CUDA:
            return cudaDivide(tensor1, tensor2);
        default:
            break;
        }

        return std::move(tensor1);
    }
} // namespace ty
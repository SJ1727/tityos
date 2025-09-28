#include "tityos/ty/ops/BinaryOps.h"

namespace ty {
    Tensor add(Tensor &tensor1, Tensor &tensor2, bool requiresGrad) {
        assertSameDevice(tensor1, tensor2);
        assertBroadcastable(tensor1.shape(), tensor2.shape());
        bool resultRequiresGrad = (tensor1.requiresGrad() || tensor2.requiresGrad()) && requiresGrad;

        Tensor result(tensor1.shape(), tensor1.dtype(), tensor1.device(), resultRequiresGrad);

        switch (tensor1.device().type()) {
        case DeviceType::CPU:
            cpuAdd(result, tensor1, tensor2);
            break;
        case DeviceType::CUDA:
            return cudaAdd(tensor1, tensor2);
        default:
            break;
        }

        return std::move(result);
    }

    Tensor subtract(Tensor &tensor1, Tensor &tensor2, bool requiresGrad) {
        assertSameDevice(tensor1, tensor2);
        assertBroadcastable(tensor1.shape(), tensor2.shape());
        bool resultRequiresGrad = (tensor1.requiresGrad() || tensor2.requiresGrad()) && requiresGrad;

        Tensor result(tensor1.shape(), tensor1.dtype(), tensor1.device(), resultRequiresGrad);

        switch (tensor1.device().type()) {
        case DeviceType::CPU:
            cpuSubtract(result, tensor1, tensor2);
            break;
        case DeviceType::CUDA:
            return cudaSubtract(tensor1, tensor2);
        default:
            break;
        }

        return std::move(result);
    }

    Tensor multiply(Tensor &tensor1, Tensor &tensor2, bool requiresGrad) {
        assertSameDevice(tensor1, tensor2);
        assertBroadcastable(tensor1.shape(), tensor2.shape());
        bool resultRequiresGrad = (tensor1.requiresGrad() || tensor2.requiresGrad()) && requiresGrad;

        Tensor result(tensor1.shape(), tensor1.dtype(), tensor1.device(), resultRequiresGrad);

        switch (tensor1.device().type()) {
        case DeviceType::CPU:
            cpuMultiply(result, tensor1, tensor2);
            break;
        case DeviceType::CUDA:
            return cudaMultiply(tensor1, tensor2);
        default:
            break;
        }

        return std::move(result);
    }

    Tensor divide(Tensor &tensor1, Tensor &tensor2, bool requiresGrad) {
        assertSameDevice(tensor1, tensor2);
        assertBroadcastable(tensor1.shape(), tensor2.shape());
        bool resultRequiresGrad = (tensor1.requiresGrad() || tensor2.requiresGrad()) && requiresGrad;

        Tensor result(tensor1.shape(), tensor1.dtype(), tensor1.device(), resultRequiresGrad);

        switch (tensor1.device().type()) {
        case DeviceType::CPU:
            cpuDivide(result, tensor1, tensor2);
            break;
        case DeviceType::CUDA:
            return cudaDivide(tensor1, tensor2);
        default:
            break;
        }

        return std::move(result);
    }
} // namespace ty
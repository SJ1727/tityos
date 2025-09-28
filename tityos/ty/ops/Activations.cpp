#include "tityos/ty/ops/Activations.h"

namespace ty {
    Tensor relu(Tensor &tensor, bool requiresGrad) {
        bool resultRequiresGrad = tensor.requiresGrad() && requiresGrad;

        Tensor result(tensor.shape(), tensor.dtype(), tensor.device(), resultRequiresGrad);

        switch (tensor.device().type()) {
        case DeviceType::CPU:
            cpuRelu(result, tensor);
            break;
        case DeviceType::CUDA:
            return cudaRelu(tensor);
        default:
            break;
        }

        return std::move(result);
    }

    Tensor step(Tensor &tensor, bool requiresGrad) {
        bool resultRequiresGrad = tensor.requiresGrad() && requiresGrad;

        Tensor result(tensor.shape(), tensor.dtype(), tensor.device(), resultRequiresGrad);

        switch (tensor.device().type()) {
        case DeviceType::CPU:
            cpuStep(result, tensor);
            break;
        case DeviceType::CUDA:
            return cudaStep(tensor);
        default:
            break;
        }

        return std::move(result);
    }
} // namespace ty
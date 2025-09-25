#include "tityos/ty/ops/Activations.h"

namespace ty {
    Tensor relu(Tensor &tensor) {
        switch (tensor.device().type()) {
        case DeviceType::CPU:
            return cpuRelu(tensor);
        case DeviceType::CUDA:
            return cudaRelu(tensor);
        default:
            break;
        }

        return std::move(tensor);
    }

    Tensor step(Tensor &tensor) {
        switch (tensor.device().type()) {
        case DeviceType::CPU:
            return cpuStep(tensor);
        case DeviceType::CUDA:
            return cudaStep(tensor);
        default:
            break;
        }

        return std::move(tensor);
    }
} // namespace ty
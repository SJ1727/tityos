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
} // namespace ty
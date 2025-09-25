#include "tityos/ty/ops/UnaryOps.h"

namespace ty {
    Tensor relu(Tensor &tensor, float pow) {
        switch (tensor.device().type()) {
        case DeviceType::CPU:
            return cpuPow(tensor, pow);
        default:
            break;
        }

        return std::move(tensor);
    }
}
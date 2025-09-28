#include "tityos/ty/ops/UnaryOps.h"

namespace ty {
    Tensor pow(Tensor &tensor, float power, bool requiresGrad) {
        bool resultRequiresGrad = tensor.requiresGrad() && requiresGrad;

        Tensor result(tensor.shape(), tensor.dtype(), tensor.device(), resultRequiresGrad);

        switch (tensor.device().type()) {
        case DeviceType::CPU:
            cpuPow(result, tensor, power);
            break;
        default:
            break;
        }

        return std::move(result);
    }
} // namespace ty
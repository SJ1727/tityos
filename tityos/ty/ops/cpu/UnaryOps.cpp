#include "tityos/ty/ops/cpu/UnaryOps.h"

namespace ty {
    void cpuPow(Tensor &result, Tensor &tensor, float power) {
        TensorIterator iter = unaryOperationIterator(result, tensor);

        /* Currently only works for floats */
        iter.forEach([=](std::vector<void *> args) {
            float *out = static_cast<float *>(args[0]);
            float *in = static_cast<float *>(args[1]);
            *out = std::pow(*in, power);
        });

        if (result.requiresGrad()) {
            result.setBackwardFunc([=](std::vector<std::shared_ptr<Tensor>> context, Tensor *grad) {
                if (context[0]->requiresGrad()) {
                    Tensor factor(std::vector<float>({power}), Shape({1}));
                    Tensor pow0 = pow(*context[0], power - 1, false);
                    Tensor mul0 = multiply(pow0, factor, false);
                    Tensor mul1 = multiply(mul0, *grad, false);
                    *context[0]->grad() = add(mul1, *context[0]->grad(), false);
                }
            });

            result.setContextTensors({std::make_shared<Tensor>(tensor)});
        }
    }
} // namespace ty
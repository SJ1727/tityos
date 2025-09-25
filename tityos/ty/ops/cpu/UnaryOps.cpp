#include "tityos/ty/ops/cpu/UnaryOps.h"

namespace ty {
    Tensor cpuPow(Tensor &tensor, float pow) {
        Tensor result(tensor.shape(), tensor.dtype());
        TensorIterator iter = unaryOperationIterator(result, tensor);

        /* Currently only works for floats */
        iter.forEach([=](std::vector<void *> args) {
            float *out = static_cast<float *>(args[0]);
            float *in = static_cast<float *>(args[1]);
            *out = std::pow(*in, pow);
        });

        if (tensor.requiresGrad()) {
            result.setBackwardFunc([=](std::vector<std::shared_ptr<Tensor>> context, Tensor *grad) {
                if (context[0]->requiresGrad()) {
                    Tensor factor(std::vector<float>({pow}), Shape({1}));
                    Tensor pow0 = cpuPow(*context[0], pow - 1);
                    Tensor mul0 = cpuMultiply(pow0, factor);
                    Tensor mul1 = cpuMultiply(mul0, *grad);
                    *context[0]->grad() = cpuAdd(mul1, *context[0]->grad());
                }
            });

            result.setContextTensors({std::make_shared<Tensor>(tensor)});
        }

        return result;
    }
} // namespace ty
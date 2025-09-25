#include "tityos/ty/ops/cpu/Activations.h"

#define TITYOS_UNARY_OP_FOREACH(TYPE, OP, iter)                                                    \
    iter.forEach([](std::vector<void *> args) {                                                    \
        TYPE *out = static_cast<TYPE *>(args[0]);                                                  \
        TYPE *in = static_cast<TYPE *>(args[1]);                                                   \
        OP;                                                                                        \
    })

namespace ty {
    Tensor cpuRelu(Tensor &tensor) {
        Tensor result(tensor.shape(), tensor.dtype());
        TensorIterator iter = unaryOperationIterator(result, tensor);

        TITYOS_TYPED_FUNC_SWITCH(
            tensor.dtype(),
            std::format("Cannot apply relu to tensor of type {}", dtypeToString(tensor.dtype())),
            TITYOS_UNARY_OP_FOREACH, *out = *in > 0 ? *in : 0, iter);

        if (tensor.requiresGrad()) {
            result.setBackwardFunc([](std::vector<std::shared_ptr<Tensor>> context, Tensor *grad) {
                if (context[0]->requiresGrad()) {
                    Tensor step0 = cpuStep(*context[0]);
                    Tensor mul0 = cpuMultiply(step0, *grad);
                    *context[0]->grad() = cpuAdd(step0, *context[0]->grad());
                }
            });

            result.setContextTensors({std::make_shared<Tensor>(tensor)});
        }

        return result;
    }

    Tensor cpuStep(Tensor &tensor) {
        Tensor result(tensor.shape(), tensor.dtype());
        TensorIterator iter = unaryOperationIterator(result, tensor);

        TITYOS_TYPED_FUNC_SWITCH(
            tensor.dtype(),
            std::format("Cannot apply step to tensor of type {}", dtypeToString(tensor.dtype())),
            TITYOS_UNARY_OP_FOREACH, *out = *in > 0 ? 1 : 0, iter);

        /* No Backward Function required*/

        return result;
    }
} // namespace ty
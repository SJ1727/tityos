#include "tityos/ty/ops/cpu/Activations.h"

#define TITYOS_UNARY_OP_FOREACH(TYPE, OP, iter)                                                    \
    iter.forEach([](std::vector<void *> args) {                                                    \
        TYPE *out = static_cast<TYPE *>(args[0]);                                                  \
        TYPE *in = static_cast<TYPE *>(args[1]);                                                   \
        OP;                                                                                        \
    })

namespace ty {
    void cpuRelu(Tensor& result, Tensor &tensor) {
        TensorIterator iter = unaryOperationIterator(result, tensor);

        TITYOS_TYPED_NUMBER_FUNC_SWITCH(
            tensor.dtype(),
            std::format("Cannot apply relu to tensor of type {}", dtypeToString(tensor.dtype())),
            TITYOS_UNARY_OP_FOREACH, *out = *in > 0 ? *in : 0, iter);

        if (result.requiresGrad()) {
            result.setBackwardFunc([](std::vector<std::shared_ptr<Tensor>> context, Tensor *grad) {
                if (context[0]->requiresGrad()) {
                    Tensor step0 = step(*context[0], false);
                    Tensor mul0 = multiply(step0, *grad, false);
                    *context[0]->grad() = add(step0, *context[0]->grad(), false);
                }
            });

            result.setContextTensors({std::make_shared<Tensor>(tensor)});
        }
    }

    void cpuStep(Tensor& result, Tensor &tensor) {
        TensorIterator iter = unaryOperationIterator(result, tensor);

        TITYOS_TYPED_NUMBER_FUNC_SWITCH(
            tensor.dtype(),
            std::format("Cannot apply step to tensor of type {}", dtypeToString(tensor.dtype())),
            TITYOS_UNARY_OP_FOREACH, *out = *in > 0 ? 1 : 0, iter);

        /* No Backward Function required*/
    }
} // namespace ty
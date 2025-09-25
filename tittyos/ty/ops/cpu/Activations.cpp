#include "tittyos/ty/ops/cpu/Activations.h"

#define tittyos_UNARY_OP_FOREACH(TYPE, OP, iter)                                                    \
    iter.forEach([](std::vector<void *> args) {                                                    \
        TYPE *out = static_cast<TYPE *>(args[0]);                                                  \
        TYPE *in = static_cast<TYPE *>(args[1]);                                                   \
        OP;                                                                                        \
    })

namespace ty {
    Tensor cpuRelu(Tensor &tensor) {
        Tensor result(tensor.shape(), tensor.dtype());
        TensorIterator iter = unaryOperationIterator(result, tensor);

        tittyos_TYPED_FUNC_SWITCH(
            tensor.dtype(),
            std::format("Cannot apply relu to tensor of type {}", dtypeToString(tensor.dtype())),
            tittyos_UNARY_OP_FOREACH, *out = *in > 0 ? *in : 0, iter);

        return result;
    }
} // namespace ty
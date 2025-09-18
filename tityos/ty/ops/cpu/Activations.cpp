#include "tityos/ty/ops/cpu/Activations.h"

#define TITYOS_UNARY_OP_FOREACH(TYPE, OP, iter)                                                    \
    iter.forEach([](std::vector<void *> args) {                                                    \
        TYPE *out = static_cast<TYPE *>(args[0]);                                                  \
        TYPE *in1 = static_cast<TYPE *>(args[1]);                                                  \
        TYPE *in2 = static_cast<TYPE *>(args[2]);                                                  \
        *out = (*in1)OP(*in2);                                                                     \
    })

namespace ty {
    Tensor cpuRelu(Tensor &tensor) {
        Tensor result(tensor.shape());
        TensorIterator iter;

        iter.unaryOperationIteration(result, tensor);

        if (tensor.dtype() == DType::float32) {
            iter.forEach([](std::vector<void *> args) {
                float *out = static_cast<float *>(args[0]);
                float *in = static_cast<float *>(args[1]);
                *out = *in > 0 ? *in : 0;
            });
        }

        return result;
    }
} // namespace ty
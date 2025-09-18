#include "tityos/ty/ops/cpu/BinaryOps.h"

namespace ty {

#define TITYOS_BINARY_OP_FOREACH(TYPE, OP, iter)                                                   \
    iter.forEach([](std::vector<void *> args) {                                                    \
        TYPE *out = static_cast<TYPE *>(args[0]);                                                  \
        TYPE *in1 = static_cast<TYPE *>(args[1]);                                                  \
        TYPE *in2 = static_cast<TYPE *>(args[2]);                                                  \
        *out = (*in1)OP(*in2);                                                                     \
    })

    Tensor cpuAdd(Tensor &tensor1, Tensor &tensor2) {
        Tensor result(ty::broadcastCombineShapes(tensor1.shape(), tensor2.shape()),
                      tensor1.dtype());
        TensorIterator iter = binaryOperationIterator(result, tensor1, tensor2);

        TITYOS_TYPED_FUNC_SWITCH(
            tensor1.dtype(),
            std::format("Cannot add tensors of type {}", dtypeToString(tensor1.dtype())),
            TITYOS_BINARY_OP_FOREACH, +, iter);

        return result;
    }

    Tensor cpuSubtract(Tensor &tensor1, Tensor &tensor2) {
        Tensor result(ty::broadcastCombineShapes(tensor1.shape(), tensor2.shape()),
                      tensor1.dtype());
        TensorIterator iter = binaryOperationIterator(result, tensor1, tensor2);

        TITYOS_TYPED_FUNC_SWITCH(
            tensor1.dtype(),
            std::format("Cannot subtract tensors of type {}", dtypeToString(tensor1.dtype())),
            TITYOS_BINARY_OP_FOREACH, -, iter);

        return result;
    }

    Tensor cpuMultiply(Tensor &tensor1, Tensor &tensor2) {
        Tensor result(ty::broadcastCombineShapes(tensor1.shape(), tensor2.shape()),
                      tensor1.dtype());
        TensorIterator iter = binaryOperationIterator(result, tensor1, tensor2);

        TITYOS_TYPED_FUNC_SWITCH(
            tensor1.dtype(),
            std::format("Cannot multiply tensors of type {}", dtypeToString(tensor1.dtype())),
            TITYOS_BINARY_OP_FOREACH, *, iter);

        return result;
    }

    Tensor cpuDivide(Tensor &tensor1, Tensor &tensor2) {
        Tensor result(ty::broadcastCombineShapes(tensor1.shape(), tensor2.shape()),
                      tensor1.dtype());
        TensorIterator iter = binaryOperationIterator(result, tensor1, tensor2);

        TITYOS_TYPED_FUNC_SWITCH(
            tensor1.dtype(),
            std::format("Cannot divide tensors of type {}", dtypeToString(tensor1.dtype())),
            TITYOS_BINARY_OP_FOREACH, /, iter);

        return result;
    }
} // namespace ty
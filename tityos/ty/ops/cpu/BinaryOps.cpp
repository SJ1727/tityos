#include "tityos/ty/ops/cpu/BinaryOps.h"

namespace ty {

#define TITYOS_BINARY_OP_FOREACH(TYPE, OP, iter)                                                   \
    iter.forEach([](std::vector<void *> args) {                                                    \
        TYPE *out = static_cast<TYPE *>(args[0]);                                                  \
        TYPE *in1 = static_cast<TYPE *>(args[1]);                                                  \
        TYPE *in2 = static_cast<TYPE *>(args[2]);                                                  \
        *out = (*in1)OP(*in2);                                                                     \
    })

#define TITYOS_BINARY_OP_FOREACH_SWITCH(OP, iter, dtype, errMessage)                               \
    switch (dtype) {                                                                               \
    case DType::float16:                                                                           \
        TITYOS_BINARY_OP_FOREACH(uint16_t, OP, iter);                                              \
        break;                                                                                     \
    case DType::float32:                                                                           \
        TITYOS_BINARY_OP_FOREACH(float, OP, iter);                                                 \
        break;                                                                                     \
    case DType::float64:                                                                           \
        TITYOS_BINARY_OP_FOREACH(double, OP, iter);                                                \
        break;                                                                                     \
    case DType::int8:                                                                              \
        TITYOS_BINARY_OP_FOREACH(int8_t, OP, iter);                                                \
        break;                                                                                     \
    case DType::int16:                                                                             \
        TITYOS_BINARY_OP_FOREACH(int16_t, OP, iter);                                               \
        break;                                                                                     \
    case DType::int32:                                                                             \
        TITYOS_BINARY_OP_FOREACH(int32_t, OP, iter);                                               \
        break;                                                                                     \
    case DType::int64:                                                                             \
        TITYOS_BINARY_OP_FOREACH(int64_t, OP, iter);                                               \
        break;                                                                                     \
    case DType::uint8:                                                                             \
        TITYOS_BINARY_OP_FOREACH(uint8_t, OP, iter);                                               \
        break;                                                                                     \
    case DType::uint16:                                                                            \
        TITYOS_BINARY_OP_FOREACH(uint16_t, OP, iter);                                              \
        break;                                                                                     \
    case DType::uint32:                                                                            \
        TITYOS_BINARY_OP_FOREACH(uint32_t, OP, iter);                                              \
        break;                                                                                     \
    case DType::uint64:                                                                            \
        TITYOS_BINARY_OP_FOREACH(uint64_t, OP, iter);                                              \
        break;                                                                                     \
    default:                                                                                       \
        throw std::runtime_error(errMessage);                                                      \
    }

    Tensor cpuAdd(Tensor &tensor1, Tensor &tensor2) {
        Tensor result(ty::broadcastCombineShapes(tensor1.shape(), tensor2.shape()),
                      tensor1.dtype());
        TensorIterator iter;

        iter.binaryOperationIteration(result, tensor1, tensor2);

        TITYOS_BINARY_OP_FOREACH_SWITCH(
            +, iter, tensor1.dtype(),
            std::format("Cannot add tensors of type {}", dtypeToString(tensor1.dtype())));

        return result;
    }

    Tensor cpuSubtract(Tensor &tensor1, Tensor &tensor2) {
        Tensor result(ty::broadcastCombineShapes(tensor1.shape(), tensor2.shape()),
                      tensor1.dtype());
        TensorIterator iter;

        iter.binaryOperationIteration(result, tensor1, tensor2);

        TITYOS_BINARY_OP_FOREACH_SWITCH(
            -, iter, tensor1.dtype(),
            std::format("Cannot subtract tensors of type {}", dtypeToString(tensor1.dtype())));

        return result;
    }

    Tensor cpuMultiply(Tensor &tensor1, Tensor &tensor2) {
        Tensor result(ty::broadcastCombineShapes(tensor1.shape(), tensor2.shape()),
                      tensor1.dtype());
        TensorIterator iter;

        iter.binaryOperationIteration(result, tensor1, tensor2);

        TITYOS_BINARY_OP_FOREACH_SWITCH(*, iter, tensor1.dtype(),
                                        std::format("Cannot multiply tensors of type {}",
                                                    dtypeToString(tensor1.dtype())));

        return result;
    }

    Tensor cpuDivide(Tensor &tensor1, Tensor &tensor2) {
        Tensor result(ty::broadcastCombineShapes(tensor1.shape(), tensor2.shape()),
                      tensor1.dtype());
        TensorIterator iter;

        iter.binaryOperationIteration(result, tensor1, tensor2);

        TITYOS_BINARY_OP_FOREACH_SWITCH(
            /, iter, tensor1.dtype(),
            std::format("Cannot divide tensors of type {}", dtypeToString(tensor1.dtype())));

        return result;
    }
} // namespace ty
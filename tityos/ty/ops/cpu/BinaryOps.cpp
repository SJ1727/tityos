#include "tityos/ty/ops/cpu/BinaryOps.h"

#include <iostream>

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

        // Addition backwards func
        if (tensor1.requiresGrad() || tensor2.requiresGrad()) {
            result.setBackwardFunc([](std::vector<std::shared_ptr<Tensor>> context, Tensor *grad) {
                if (context[0]->requiresGrad()) {
                    *context[0]->grad() = cpuAdd(*grad, *context[0]->grad());
                }

                if (context[1]->requiresGrad()) {
                    *context[1]->grad() = cpuAdd(*grad, *context[1]->grad());
                }
            });

            result.setContextTensors(
                {std::make_shared<Tensor>(tensor1), std::make_shared<Tensor>(tensor2)});
        }

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

        // Subtraction backwards func
        if (tensor1.requiresGrad() || tensor2.requiresGrad()) {
            result.setBackwardFunc([](std::vector<std::shared_ptr<Tensor>> context, Tensor *grad) {
                if (context[0]->requiresGrad()) {
                    *context[0]->grad() = cpuAdd(*grad, *context[0]->grad());
                }

                if (context[1]->requiresGrad()) {
                    *context[1]->grad() = cpuSubtract(*context[1]->grad(), *grad);
                }
            });

            result.setContextTensors(
                {std::make_shared<Tensor>(tensor1), std::make_shared<Tensor>(tensor2)});
        }

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

        // Multiplication backwards func
        if (tensor1.requiresGrad() || tensor2.requiresGrad()) {
            result.setBackwardFunc([](std::vector<std::shared_ptr<Tensor>> context, Tensor *grad) {
                if (context[0]->requiresGrad()) {
                    Tensor mul0 = cpuMultiply(*context[1], *grad);
                    *context[0]->grad() = cpuAdd(mul0, *context[0]->grad());
                }

                if (context[1]->requiresGrad()) {
                    Tensor mul1 = cpuMultiply(*context[0], *grad);
                    *context[1]->grad() = cpuAdd(mul1, *context[1]->grad());
                }
            });

            result.setContextTensors(
                {std::make_shared<Tensor>(tensor1), std::make_shared<Tensor>(tensor2)});
        }

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

        // Division backwards func
        if (tensor1.requiresGrad() || tensor2.requiresGrad()) {
            result.setBackwardFunc([](std::vector<std::shared_ptr<Tensor>> context, Tensor *grad) {
                if (context[0]->requiresGrad()) {
                    Tensor pow0 = cpuPow(*context[1], -1);
                    Tensor mul0 = cpuMultiply(pow0, *grad);
                    *context[0]->grad() = cpuAdd(mul0, *context[0]->grad());
                }

                if (context[1]->requiresGrad()) {
                    Tensor pow0 = cpuPow(*context[1], -2);
                    Tensor mul1 = cpuMultiply(*context[0], pow0);
                    Tensor mul2 = cpuMultiply(mul1, *grad);
                    *context[1]->grad() = cpuSubtract(*context[1]->grad(), mul2);
                }
            });

            result.setContextTensors(
                {std::make_shared<Tensor>(tensor1), std::make_shared<Tensor>(tensor2)});
        }

        return result;
    }
} // namespace ty
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

    void cpuAdd(Tensor &result, Tensor &tensor1, Tensor &tensor2) {
        TensorIterator iter = binaryOperationIterator(result, tensor1, tensor2);

        TITYOS_TYPED_NUMBER_FUNC_SWITCH(
            tensor1.dtype(),
            std::format("Cannot add tensors of type {}", dtypeToString(tensor1.dtype())),
            TITYOS_BINARY_OP_FOREACH, +, iter);

        // Addition backwards func
        if (result.requiresGrad()) {
            result.setBackwardFunc([](std::vector<std::shared_ptr<Tensor>> context, Tensor *grad) {
                if (context[0]->requiresGrad()) {
                    *context[0]->grad() = add(*grad, *context[0]->grad(), false);
                }

                if (context[1]->requiresGrad()) {
                    *context[1]->grad() = add(*grad, *context[1]->grad(), false);
                }
            });

            result.setContextTensors(
                {std::make_shared<Tensor>(tensor1), std::make_shared<Tensor>(tensor2)});
        }
    }

    void cpuSubtract(Tensor &result, Tensor &tensor1, Tensor &tensor2) {
        TensorIterator iter = binaryOperationIterator(result, tensor1, tensor2);

        TITYOS_TYPED_NUMBER_FUNC_SWITCH(
            tensor1.dtype(),
            std::format("Cannot subtract tensors of type {}", dtypeToString(tensor1.dtype())),
            TITYOS_BINARY_OP_FOREACH, -, iter);

        // Subtraction backwards func
        if (result.requiresGrad()) {
            result.setBackwardFunc([](std::vector<std::shared_ptr<Tensor>> context, Tensor *grad) {
                if (context[0]->requiresGrad()) {
                    *context[0]->grad() = add(*grad, *context[0]->grad(), false);
                }

                if (context[1]->requiresGrad()) {
                    *context[1]->grad() = subtract(*context[1]->grad(), *grad, false);
                }
            });

            result.setContextTensors(
                {std::make_shared<Tensor>(tensor1), std::make_shared<Tensor>(tensor2)});
        }
    }

    void cpuMultiply(Tensor &result, Tensor &tensor1, Tensor &tensor2) {
        TensorIterator iter = binaryOperationIterator(result, tensor1, tensor2);

        TITYOS_TYPED_NUMBER_FUNC_SWITCH(
            tensor1.dtype(),
            std::format("Cannot multiply tensors of type {}", dtypeToString(tensor1.dtype())),
            TITYOS_BINARY_OP_FOREACH, *, iter);

        // Multiplication backwards func
        if (result.requiresGrad()) {
            result.setBackwardFunc([](std::vector<std::shared_ptr<Tensor>> context, Tensor *grad) {
                if (context[0]->requiresGrad()) {
                    Tensor mul0 = multiply(*context[1], *grad, false);
                    *context[0]->grad() = add(mul0, *context[0]->grad(), false);
                }

                if (context[1]->requiresGrad()) {
                    Tensor mul1 = multiply(*context[0], *grad, false);
                    *context[1]->grad() = add(mul1, *context[1]->grad(), false);
                }
            });

            result.setContextTensors(
                {std::make_shared<Tensor>(tensor1), std::make_shared<Tensor>(tensor2)});
        }
    }

    void cpuDivide(Tensor &result, Tensor &tensor1, Tensor &tensor2) {
        TensorIterator iter = binaryOperationIterator(result, tensor1, tensor2);

        TITYOS_TYPED_NUMBER_FUNC_SWITCH(
            tensor1.dtype(),
            std::format("Cannot divide tensors of type {}", dtypeToString(tensor1.dtype())),
            TITYOS_BINARY_OP_FOREACH, /, iter);

        // Division backwards func
        if (result.requiresGrad()) {
            result.setBackwardFunc([](std::vector<std::shared_ptr<Tensor>> context, Tensor *grad) {
                if (context[0]->requiresGrad()) {
                    Tensor pow0 = pow(*context[1], -1, false);
                    Tensor mul0 = multiply(pow0, *grad, false);
                    *context[0]->grad() = add(mul0, *context[0]->grad(), false);
                }

                if (context[1]->requiresGrad()) {
                    Tensor pow0 = pow(*context[1], -2, false);
                    Tensor mul1 = multiply(*context[0], pow0, false);
                    Tensor mul2 = multiply(mul1, *grad, false);
                    *context[1]->grad() = subtract(*context[1]->grad(), mul2, false);
                }
            });

            result.setContextTensors(
                {std::make_shared<Tensor>(tensor1), std::make_shared<Tensor>(tensor2)});
        }
    }
} // namespace ty
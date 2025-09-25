#include "tittyos/ty/ops/cuda/BinaryOps.h"

namespace ty {
    Tensor cudaAdd(Tensor &tensor1, Tensor &tensor2) {
        return std::move(tensor1);
    }

    Tensor cudaSubtract(Tensor &tensor1, Tensor &tensor2) {
        return std::move(tensor1);
    }

    Tensor cudaMultiply(Tensor &tensor1, Tensor &tensor2) {
        return std::move(tensor1);
    }

    Tensor cudaDivide(Tensor &tensor1, Tensor &tensor2) {
        return std::move(tensor1);
    }
} // namespace ty
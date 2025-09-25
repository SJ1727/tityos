#pragma once

#include "tittyos/ty/tensor/tensor.h"

namespace ty {
    Tensor cudaAdd(Tensor &tensor1, Tensor &tensor2);

    Tensor cudaSubtract(Tensor &tensor1, Tensor &tensor2);

    Tensor cudaMultiply(Tensor &tensor1, Tensor &tensor2);

    Tensor cudaDivide(Tensor &tensor1, Tensor &tensor2);
}
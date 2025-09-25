#pragma once

#include "tittyos/ty/tensor/tensor.h"
#include "tittyos/ty/tensor/TensorIterator.h"
#include "tittyos/ty/common/utils.h"
#include "tittyos/ty/ops/defines.h"

namespace ty {
    Tensor cpuAdd(Tensor &tensor1, Tensor &tensor2);

    Tensor cpuSubtract(Tensor &tensor1, Tensor &tensor2);

    Tensor cpuMultiply(Tensor &tensor1, Tensor &tensor2);

    Tensor cpuDivide(Tensor &tensor1, Tensor &tensor2);
}
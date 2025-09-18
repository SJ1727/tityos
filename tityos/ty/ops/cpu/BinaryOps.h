#pragma once

#include "tityos/ty/tensor/tensor.h"
#include "tityos/ty/tensor/TensorIterator.h"
#include "tityos/ty/common/utils.h"
#include "tityos/ty/ops/defines.h"

namespace ty {
    Tensor cpuAdd(Tensor &tensor1, Tensor &tensor2);

    Tensor cpuSubtract(Tensor &tensor1, Tensor &tensor2);

    Tensor cpuMultiply(Tensor &tensor1, Tensor &tensor2);

    Tensor cpuDivide(Tensor &tensor1, Tensor &tensor2);
}
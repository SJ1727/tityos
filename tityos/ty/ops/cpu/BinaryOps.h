#pragma once

#include "tityos/ty/common/utils.h"
#include "tityos/ty/ops/UnaryOps.h"
#include "tityos/ty/ops/BinaryOps.h"
#include "tityos/ty/ops/defines.h"
#include "tityos/ty/tensor/TensorIterator.h"
#include "tityos/ty/tensor/tensor.h"
#include "tityos/ty/utils/utils.h"

namespace ty {
    void cpuAdd(Tensor &result, Tensor &tensor1, Tensor &tensor2);

    void cpuSubtract(Tensor &result, Tensor &tensor1, Tensor &tensor2);

    void cpuMultiply(Tensor &result, Tensor &tensor1, Tensor &tensor2);

    void cpuDivide(Tensor &result, Tensor &tensor1, Tensor &tensor2);
} // namespace ty
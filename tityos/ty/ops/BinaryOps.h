#pragma once

#include <stdexcept>
#include <format>

#include "tityos/ty/ops/cpu/BinaryOps.h"
#include "tityos/ty/ops/cuda/BinaryOps.h"

#include "tityos/ty/common/assert.h"
#include "tityos/ty/tensor/tensor.h"

namespace ty {
    Tensor add(Tensor &tensor1, Tensor &tensor2);

    Tensor subtract(Tensor &tensor1, Tensor &tensor2);

    Tensor multiply(Tensor &tensor1, Tensor &tensor2);

    Tensor divide(Tensor &tensor1, Tensor &tensor2);
}
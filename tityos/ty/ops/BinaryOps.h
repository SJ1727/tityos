#pragma once

#include <stdexcept>
#include <format>

#include "tityos/ty/common/defines.h"
#include "tityos/ty/common/assert.h"
#include "tityos/ty/tensor/tensor.h"

#include "tityos/ty/ops/cpu/BinaryOps.h"
#include "tityos/ty/ops/cuda/BinaryOps.h"

namespace ty {
    Tensor TITYOS_API add(Tensor &tensor1, Tensor &tensor2);

    Tensor TITYOS_API subtract(Tensor &tensor1, Tensor &tensor2);

    Tensor TITYOS_API multiply(Tensor &tensor1, Tensor &tensor2);

    Tensor TITYOS_API divide(Tensor &tensor1, Tensor &tensor2);
}
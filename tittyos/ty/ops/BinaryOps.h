#pragma once

#include <stdexcept>
#include <format>

#include "tittyos/ty/common/defines.h"
#include "tittyos/ty/common/assert.h"
#include "tittyos/ty/tensor/tensor.h"

#include "tittyos/ty/ops/cpu/BinaryOps.h"
#include "tittyos/ty/ops/cuda/BinaryOps.h"

namespace ty {
    Tensor tittyos_API add(Tensor &tensor1, Tensor &tensor2);

    Tensor tittyos_API subtract(Tensor &tensor1, Tensor &tensor2);

    Tensor tittyos_API multiply(Tensor &tensor1, Tensor &tensor2);

    Tensor tittyos_API divide(Tensor &tensor1, Tensor &tensor2);
}
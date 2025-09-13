#pragma once

#include "tityos/ty/ops/cpu/Addition.h"
#include "tityos/ty/ops/cuda/Addition.h"

#include "tityos/ty/common/assert.h"
#include "tityos/ty/tensor/tensor.h"

namespace ty {
    Tensor add(Tensor &tensor1, Tensor &tensor2);
}
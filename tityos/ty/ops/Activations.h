#pragma once

#include "tityos/ty/common/defines.h"
#include "tityos/ty/tensor/TensorIterator.h"
#include "tityos/ty/tensor/tensor.h"

#include "tityos/ty/ops/cpu/Activations.h"
#include "tityos/ty/ops/cuda/Activations.h"

namespace ty {
    Tensor TITYOS_API relu(Tensor &tensor, bool requiresGrad = true);

    Tensor TITYOS_API step(Tensor &tensor, bool requiresGrad = true);
}
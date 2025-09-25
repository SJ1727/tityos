#pragma once

#include "tittyos/ty/common/defines.h"
#include "tittyos/ty/tensor/TensorIterator.h"
#include "tittyos/ty/tensor/tensor.h"

#include "tittyos/ty/ops/cpu/Activations.h"
#include "tittyos/ty/ops/cuda/Activations.h"

namespace ty {
    Tensor tittyos_API relu(Tensor &tensor);
}
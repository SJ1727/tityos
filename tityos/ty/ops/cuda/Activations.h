#pragma once

#include "tityos/ty/tensor/tensor.h"

namespace ty {
    Tensor cudaRelu(Tensor &tensor);

    Tensor cudaStep(Tensor &tensor);
}
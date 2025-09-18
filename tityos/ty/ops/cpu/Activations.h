#pragma once

#include "tityos/ty/tensor/tensor.h"
#include "tityos/ty/tensor/TensorIterator.h"
#include "tityos/ty/common/utils.h"

namespace ty {
    Tensor cpuRelu(Tensor& tensor);
}
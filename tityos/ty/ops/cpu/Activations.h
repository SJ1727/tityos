#pragma once

#include "tityos/ty/tensor/tensor.h"
#include "tityos/ty/tensor/TensorIterator.h"
#include "tityos/ty/common/utils.h"
#include "tityos/ty/ops/defines.h"
#include "tityos/ty/ops/cpu/BinaryOps.h"
#include "tityos/ty/utils/utils.h"

namespace ty {
    Tensor cpuRelu(Tensor& tensor);

    Tensor cpuStep(Tensor& tensor);
}
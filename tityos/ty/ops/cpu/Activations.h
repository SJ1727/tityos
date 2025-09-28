#pragma once

#include "tityos/ty/tensor/tensor.h"
#include "tityos/ty/tensor/TensorIterator.h"
#include "tityos/ty/common/utils.h"
#include "tityos/ty/ops/defines.h"
#include "tityos/ty/ops/BinaryOps.h"
#include "tityos/ty/ops/Activations.h"
#include "tityos/ty/utils/utils.h"

namespace ty {
    void cpuRelu(Tensor& result, Tensor& tensor);

    void cpuStep(Tensor& result, Tensor& tensor);
}
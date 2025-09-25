#pragma once

#include "tittyos/ty/tensor/tensor.h"
#include "tittyos/ty/tensor/TensorIterator.h"
#include "tittyos/ty/common/utils.h"
#include "tittyos/ty/ops/defines.h"

namespace ty {
    Tensor cpuRelu(Tensor& tensor);
}
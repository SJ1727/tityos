#pragma once

#include "tityos/ty/tensor/tensor.h"

#include "tityos/ty/ops/cpu/UnaryOps.h"

namespace ty {
    Tensor pow(Tensor& tensor, float power, bool requiresGrad = true);
}
#pragma once

#include <cmath>

#include "tityos/ty/tensor/tensor.h"
#include "tityos/ty/ops/cpu/BinaryOps.h"

namespace ty {
    Tensor cpuPow(Tensor& tensor, float pow);
}
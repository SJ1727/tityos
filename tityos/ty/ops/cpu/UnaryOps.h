#pragma once

#include <cmath>

#include "tityos/ty/tensor/tensor.h"
#include "tityos/ty/ops/BinaryOps.h"
#include "tityos/ty/ops/UnaryOps.h"

namespace ty {
    void cpuPow(Tensor& result, Tensor& tensor, float power);
}
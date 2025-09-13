#pragma once

#include <stdexcept>
#include <format>

#include "tityos/ty/tensor/tensor.h"

namespace ty {
    void assertSameDevice(Tensor &tensor1, Tensor &tensor2);
}
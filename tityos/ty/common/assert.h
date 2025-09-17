#pragma once

#include <format>
#include <stdexcept>

#include "tityos/ty/tensor/tensor.h"

namespace ty {
    void assertSameDevice(const Tensor &tensor1, const Tensor &tensor2);

    void assertBroadcastable(const Shape &shape1, const Shape &shape2);
}; // namespace ty
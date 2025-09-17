#pragma once

#include "tityos/ty/common/assert.h"
#include "tityos/ty/tensor/tensor.h"

namespace ty {
    Shape broadcastCombineShapes(const Shape &shape1, const Shape &shape2);
}
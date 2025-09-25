#pragma once

#include "tittyos/ty/common/assert.h"
#include "tittyos/ty/tensor/tensor.h"

namespace ty {
    Shape broadcastCombineShapes(const Shape &shape1, const Shape &shape2);
}
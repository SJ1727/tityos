#pragma once

#include "tityos/ty/tensor/tensor.h"
#include "tityos/ty/utils/defines.h"

namespace ty {
    Tensor onesLike(const Tensor& tensor);

    Tensor zerosLike(const Tensor& tensor);
}
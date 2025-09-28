#pragma once

#include "tityos/ty/tensor/tensor.h"
#include "tityos/ty/common/defines.h"

namespace ty {
    TITYOS_API Tensor onesLike(const Tensor& tensor);

    TITYOS_API Tensor zerosLike(const Tensor& tensor);
}
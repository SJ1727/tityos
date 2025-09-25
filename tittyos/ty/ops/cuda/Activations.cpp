#include "tittyos/ty/ops/cuda/Activations.h"

namespace ty {
    Tensor cudaRelu(Tensor &tensor) {
        return std::move(tensor);
    }
} // namespace ty
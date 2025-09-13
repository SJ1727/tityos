#include "tityos/ty/common/assert.h"

namespace ty {
    void assertSameDevice(const Tensor &tensor1, const Tensor &tensor2) {
        if (tensor1.device() != tensor2.device()) {
            throw std::runtime_error(
                std::format("Cannot perform operation on tensors on different devices ({} and {})",
                            deviceToString(tensor1.device()), deviceToString(tensor2.device())));
        }
    }
} // namespace ty
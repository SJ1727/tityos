#include "tittyos/ty/common/assert.h"

namespace ty {
    void assertSameDevice(const Tensor &tensor1, const Tensor &tensor2) {
        if (tensor1.device() != tensor2.device()) {
            throw std::runtime_error(
                std::format("Cannot perform operation on tensors on different devices ({} and {})",
                            deviceToString(tensor1.device()), deviceToString(tensor2.device())));
        }
    }

    void assertBroadcastable(const Shape &shape1, const Shape &shape2) {
        int shapeDifference = abs(shape1.numDims() - shape2.numDims());
        int minDims = shape1.numDims() < shape2.numDims() ? shape1.numDims() : shape2.numDims();

        // Checking if shape of other shape is suitable for broadcasting
        for (int i = 0; i < minDims; i++) {
            if (shape1[shape1.numDims() - i - 1] != shape2[shape2.numDims() - i - 1] &&
                shape1[shape1.numDims() - i - 1] != 1 && shape2[shape2.numDims() - i - 1] != 1) {
                throw std::runtime_error(std::format(
                    "Shape mismatch: expected {} or 1, but got {}.",
                    shape1[shape1.numDims() - i - 1], shape2[shape2.numDims() - i - 1]));
            }
        }
    }
}; // namespace ty
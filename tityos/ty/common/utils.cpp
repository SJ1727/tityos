#include "tityos/ty/common/utils.h"

namespace ty {
    Shape broadcastCombineShapes(const Shape &shape1, const Shape &shape2) {
        assertBroadcastable(shape1, shape2);

        int minDims = shape1.numDims() < shape2.numDims() ? shape1.numDims() : shape2.numDims();
        int maxDims = shape1.numDims() > shape2.numDims() ? shape1.numDims() : shape2.numDims();
        std::vector<int64_t> newShape(maxDims, 1);

        // Uses the max of the two shapes dimensions
        for (int i = 0; i < minDims; i++) {
            newShape[maxDims - i - 1] =
                shape1[shape1.numDims() - i - 1] > shape2[shape2.numDims() - i - 1]
                    ? shape1[shape1.numDims() - i - 1]
                    : shape2[shape2.numDims() - i - 1];
        }

        // Fills in remaining dimensions
        for (int i = 0; i < shape1.numDims() - minDims; i++) {
            newShape[i] = shape1[i];
        }
        for (int i = 0; i < shape2.numDims() - minDims; i++) {
            newShape[i] = shape2[i];
        }

        return Shape(newShape);
    }
} // namespace Tityos
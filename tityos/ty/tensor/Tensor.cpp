#include "tityos/ty/tensor/Tensor.h"

#include <iostream>

namespace ty {
    Tensor::Tensor(Shape shape, DType dtype) : shape_(shape), offset_(0) {
        calculateStrides();

        dataStorage_ = std::make_shared<Storage>(shape_.numElements() * dtypeSize(dtype), dtype);
    }

    void Tensor::calculateStrides() {
        int64_t numElements = 1;
        for (int i = shape_.numDims() - 1; i >= 0; i--) {
            strides_[i] = numElements;
            numElements *= shape_[i];
        }
    }
} // namespace ty

#include "tityos/ty/tensor/Shape.h"

namespace ty {
    Shape::Shape() {
        dims_.fill(-1);
        numDims_ = 0;
    }

    Shape::Shape(std::initializer_list<int64_t> dims) {
        dims_.fill(-1);
        numDims_ = 0;

        for (int64_t dim : dims) {
            dims_[numDims_] = dim;
            numDims_++;
        }
    }

    int64_t &Shape::operator[](size_t index) {
        if (index >= numDims_) {
            throw std::out_of_range("Shape index out of range");
        }

        return dims_[index];
    }

    const int64_t &Shape::operator[](size_t index) const {
        if (index >= numDims_) {
            throw std::out_of_range("Shape index out of range");
        }

        return dims_[index];
    }

    void Shape::operator=(const Shape &shape) {
        numDims_ = shape.numDims_;
        dims_ = shape.dims_;
    }

    void Shape::operator=(Shape &&shape) {
        numDims_ = shape.numDims_;
        dims_ = std::move(shape.dims_);
    }
}; // namespace ty
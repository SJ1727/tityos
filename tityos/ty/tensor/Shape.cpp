#include "tityos/ty/tensor/Shape.h"

namespace ty {

    Shape::Shape(std::vector<int> dims) : dims_(dims) {
        if (dims_.size() < 1) {
            throw std::invalid_argument("Cannot create shape with zero dims");
        }
    }

    Shape::Shape(std::initializer_list<int> dims) : dims_(dims) {
        if (dims_.size() < 1) {
            throw std::invalid_argument("Cannot create shape with zero dims");
        }
    }

    int Shape::numDims() const {
        return dims_.size();
    }

    int Shape::numElements() const {
        return std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
    }

    int Shape::dim(const size_t dim) const {
        return dims_[dim];
    }

    int &Shape::operator[](const size_t dim) {
        return dims_[dim];
    }

    bool Shape::operator==(const Shape &other) const {
        if (other.numDims() != this->numDims()) {
            return false;
        }

        for (size_t i = 0; i < other.numDims(); i++) {
            if (this->dim(i) != other.dim(i)) {
                return false;
            }
        }

        return true;
    }

    bool Shape::operator==(const std::initializer_list<int> &other) const {
        return dims_ == std::vector<int>(other);
    }

} // namespace ty
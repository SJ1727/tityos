#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ty {
    constexpr size_t tensorMaxDims = 64;

    class Shape {
      private:
        std::array<int64_t, tensorMaxDims> dims_;
        size_t numDims_;

      public:
        Shape();

        Shape(const Shape &shape) : dims_(shape.dims_), numDims_(shape.numDims_) {}

        Shape(std::initializer_list<int64_t> dims);

        Shape(const std::vector<int64_t> &dims);

        ~Shape() = default;

        inline size_t numDims() const {
            return numDims_;
        }

        inline int64_t numElements() const {
            return std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
        }

        inline const std::array<int64_t, tensorMaxDims>& array() const {
            return dims_;
        }

        int64_t &operator[](size_t index);
        const int64_t &operator[](size_t index) const;

        void operator=(const Shape &shape);
        void operator=(Shape &&shape);
    };
}; // namespace ty
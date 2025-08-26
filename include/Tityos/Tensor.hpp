#pragma once

#include <algorithm>
#include <compare>
#include <format>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "Tityos/Slice.hpp"
#include "utils.hpp"

namespace Tityos {

    template <typename T> class Tensor {
      public:
        Tensor(const std::vector<int> &shape)
            : Tensor(std::vector<T>(vectorElementProduct(shape), 0.0f), shape) {}
        Tensor(std::vector<T> data, const std::vector<int> &shape);
        Tensor(std::shared_ptr<std::vector<T>> data, const std::vector<int> &strides,
               const std::vector<int> &shape, int offset);
        virtual ~Tensor() = default;

        int size() const;
        int numDims() const;
        std::vector<int> shape() const;
        void print() const;

        Tensor<T> at(std::vector<Slice> slices) const;
        T item() const;

        bool isContiguous() const;
        Tensor<T> contiguous() const;
        Tensor<T> clone() const;

      protected:
        int tensorIndexToFlat(std::vector<int> index) const;
        void printRecurse(int dim, std::vector<int> idx) const;

      protected:
        std::vector<int> shape_;
        int offset_;
        std::vector<int> strides_;
        std::shared_ptr<std::vector<T>> data_;
    };
} // namespace Tityos
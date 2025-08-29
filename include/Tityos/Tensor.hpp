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
        std::shared_ptr<std::vector<T>> data() const;

        Tensor<T> at(const std::vector<Slice> &slices) const;
        T item() const;

        bool isContiguous() const;
        Tensor<T> contiguous() const;
        Tensor<T> clone() const;

        void permute(const std::vector<int> &dims);
        void transpose(int dim1, int dim2);
        void transpose();

        Tensor<T> reshape(const std::vector<int> &newShape);

      protected:
        std::vector<T> getDataFlat() const;
        int getFlatIndex(const std::vector<int> &index) const;
        void printRecurse(int dim, std::vector<int> idx) const;

      protected:
        std::vector<int> shape_;
        int offset_;
        std::vector<int> strides_;
        std::shared_ptr<std::vector<T>> data_;
    };
} // namespace Tityos
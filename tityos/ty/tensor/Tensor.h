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

#include "tityos/ty/common/utils.h"
#include "tityos/ty/tensor/Index.h"
#include "tityos/ty/tensor/Shape.h"

namespace ty {

    template <typename T> class Tensor {
      public:
        Tensor(const Shape &shape)
            : Tensor(std::vector<T>(shape.numElements(), 0.0f), shape) {}
        Tensor(std::vector<T> data, const Shape &shape);
        Tensor(std::shared_ptr<std::vector<T>> data, const std::vector<int> &strides,
               const Shape &shape, int offset);
        Tensor(T value) : Tensor({value}, Shape({1})) {}
        virtual ~Tensor() = default;

        Shape shape() const;
        void print() const;
        std::shared_ptr<std::vector<T>> data() const;

        Tensor<T> at(const std::vector<Index> &slices) const;
        T itemAt(const std::vector<int> &index) const;
        T item() const;

        bool isContiguous() const;
        Tensor<T> contiguous() const;
        Tensor<T> clone() const;

        void permute(const std::vector<int> &dims);
        void transpose(int dim1, int dim2);
        void transpose();

        Tensor<T> reshape(const Shape &newShape);

        void operator=(const Tensor<T> &other);

      protected:
        std::vector<T> getDataFlat() const;
        int getFlatIndex(const std::vector<int> &index) const;
        void printRecurse(int dim, std::vector<int> idx) const;

      protected:
        Shape shape_;
        int offset_;
        std::vector<int> strides_;
        std::shared_ptr<std::vector<T>> data_;
    };
} // namespace ty
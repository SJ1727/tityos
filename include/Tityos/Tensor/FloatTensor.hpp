#pragma once

#include "Tityos/Tensor/TensorBase.hpp"

namespace Tityos {
    namespace Tensor {
        class FloatTensor : public TensorBase {
          public:
            FloatTensor(const std::vector<int> &shape)
                : FloatTensor(std::vector<float>(vectorElementProduct(shape), 0.0f), shape) {}
            FloatTensor(std::vector<float> data, const std::vector<int> &shape);
            FloatTensor(std::shared_ptr<std::vector<float>> data, const std::vector<int> &strides,
                        const std::vector<int> &shape, int offset);

            int size() const;
            int numDims() const;
            std::vector<int> shape() const override;
            void print() const override;

            FloatTensor at(std::vector<Tityos::Tensor::Slice> slices) const;
            float item() const;

            bool isContiguous() const;
            FloatTensor contiguous() const;
            FloatTensor clone() const;

          private:
            int tensorIndexToFlat(std::vector<int> index) const;
            void printRecurse(int dim, std::vector<int> idx) const;

          private:
            std::vector<int> shape_;
            int offset_;
            std::vector<int> strides_;
            std::shared_ptr<std::vector<float>> data_;
        };
    } // namespace Tensor
} // namespace Tityos
#pragma once

#include "Tityos/Tensor/TensorBase.hpp"

namespace Tityos {
    namespace Tensor {
        class FloatTensor : public TensorBase {
          public:
            FloatTensor(const std::vector<int> &shape);
            FloatTensor(const std::vector<int> &shape, std::vector<float> data);
            FloatTensor(std::shared_ptr<std::vector<float>> data, const std::vector<int> &dataShape,
                        const std::vector<int> &shape, int offset);
            std::vector<int> shape() const override;
            void print() const override;

            FloatTensor at(std::vector<Tityos::Tensor::Slice> slices) const;
            float item() const;

          private:
            int tensorIndexToFlat(std::vector<int> index) const;
            void printRecurse(int dim, std::vector<int> idx) const;

          private:
            std::vector<int> shape_;
            int offset_;
            std::vector<int> dataShape_;
            std::shared_ptr<std::vector<float>> data_;
        };
    } // namespace Tensor
} // namespace Tityos
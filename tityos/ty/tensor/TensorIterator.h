#pragma once

#include <functional>
#include <vector>

#include "tityos/ty/tensor/Tensor.h"

namespace ty {
    enum class OperandType { INPUT, OUTPUT };

    struct OperandInfo {
        Tensor &tensor;
        OperandType type;
    };

    class TensorIterator {
      private:
        std::vector<OperandInfo> operands_;

      public:
        TensorIterator() = default;

        ~TensorIterator() = default;

        void addOutput(Tensor &tensor) {
            operands_.push_back(OperandInfo({tensor, OperandType::OUTPUT}));
        }

        void addInput(Tensor &tensor) {
            operands_.push_back(OperandInfo({tensor, OperandType::INPUT}));
        }

        int64_t numElements() const {
            // Output is first element by default
            // TODO: Change this
            return operands_[0].tensor.shape().numElements();
        }

        void forEach(std::function<void(std::vector<void *> &)> func);
    };

    TensorIterator unaryOperationIterator(Tensor &out, Tensor &in);
    TensorIterator binaryOperationIterator(Tensor &out, Tensor &in1, Tensor &in2);
}; // namespace ty
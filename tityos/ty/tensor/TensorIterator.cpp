#include "tityos/ty/tensor/TensorIterator.h"

namespace ty {
    void TensorIterator::unaryOperationIteration(Tensor &out, Tensor &in) {
        this->addOutput(out);
        this->addInput(in);
    }

    void TensorIterator::binaryOperationIteration(Tensor &out, Tensor &in1, Tensor &in2) {
        this->addOutput(out);
        this->addInput(in1);
        this->addInput(in2);
    }

    void TensorIterator::forEach(std::function<void(std::vector<void *> &)> func) {
        size_t idx = 0;
        int64_t total = this->numElements();
        std::array<int64_t, tensorMaxDims> iterIndex;
        std::vector<void *> args;
        iterIndex.fill(0);
        args.resize(operands_.size());

        for (int64_t i = 0; i < total; i++) {
            // Calculating the pointer to the correct element in each tensor (input and output)
            for (int j = 0; j < args.size(); j++) {
                idx = operands_[j].tensor.offset();

                for (int k = 0; k < operands_[j].tensor.numDims(); k++) {
                    if (operands_[j].tensor.shape().array()[k] != 1) {
                        idx += iterIndex[k] * operands_[j].tensor.strides()[k];
                    }
                }

                args[j] = operands_[j].tensor.get(idx);
            }

            // Apply the provided function to the calculated pointers
            func(args);

            // Increment the index
            for (int j = 0; j < tensorMaxDims; j++) {
                if (iterIndex[j] + 1 >= operands_[0].tensor.shape().array()[j]) {
                    iterIndex[j] = 0;
                } else {
                    iterIndex[j]++;
                    break;
                }
            }
        }
    }
}; // namespace ty
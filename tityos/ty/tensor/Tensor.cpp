#include "tityos/ty/tensor/Tensor.h"

#include <iostream>

namespace ty {
    Tensor::Tensor(Shape shape, DType dtype, bool requiresGrad)
        : shape_(shape), offset_(0), requiresGrad_(requiresGrad) {
        calculateStrides();

        dataStorage_ = std::make_shared<Storage>(shape_.numElements() * dtypeSize(dtype), dtype);

        if (requiresGrad_) {
            this->zeroGrad();
        }
    }

    void Tensor::calculateStrides() {
        int64_t numElements = 1;
        for (int i = shape_.numDims() - 1; i >= 0; i--) {
            strides_[i] = numElements;
            numElements *= shape_[i];
        }
    }

    void Tensor::setRequiresGrad(bool requiresGrad) {
        if (!dtypeSupportsGrad(dataStorage_->dtype()) && requiresGrad) {
            throw std::runtime_error(std::format("Tensors of type {} do not support gradients",
                                                 dtypeToString(dataStorage_->dtype())));
        }

        requiresGrad_ = requiresGrad;
    }

    void Tensor::backward(std::shared_ptr<Tensor> initialGrad) {
        grad_ = initialGrad;

        std::unordered_map<Tensor *, bool> nodesVisited;
        std::unordered_map<Tensor *, bool> inStack;
        std::stack<std::pair<Tensor *, int>> nodeSortStack;

        nodeSortStack.push({this, 0});
        nodesVisited[this] = true;

        while (!nodeSortStack.empty()) {
            auto &[currNode, idx] = nodeSortStack.top();
            auto context = currNode->contextTensors_;

            if (idx < (int)context.size()) {
                Tensor *parentNode = context[idx++].get();
                if (!nodesVisited[parentNode]) {
                    nodesVisited[parentNode] = true;
                    nodeSortStack.push({parentNode, 0});
                }
            } else {
                if (currNode->backwardFunc_ != nullptr) {
                    currNode->backwardFunc_(context, currNode->grad().get());
                }
                nodeSortStack.pop();
            }
        }
    }
} // namespace ty

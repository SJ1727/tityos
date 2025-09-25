#include "tityos/ty/utils/TensorCreators.h"

#define TITYOS_CREATE_TENSOR_WITH_SINGLE_VALUE(TYPE, inTensor, value)                              \
    likeTensor = Tensor(std::vector<TYPE>(inTensor.numElements(), static_cast<TYPE>(value)),       \
                        inTensor.shape(), inTensor.dtype(), inTensor.device());

namespace ty {
    Tensor onesLike(const Tensor &tensor) {
        Tensor likeTensor;

        TITYOS_TYPED_BOOL_FUNC_SWITCH(tensor.dtype(), "", TITYOS_CREATE_TENSOR_WITH_SINGLE_VALUE, tensor,
                                 1);

        return likeTensor;
    }

    Tensor zerosLike(const Tensor &tensor) {
        Tensor likeTensor;

        TITYOS_TYPED_BOOL_FUNC_SWITCH(tensor.dtype(), "", TITYOS_CREATE_TENSOR_WITH_SINGLE_VALUE, tensor,
                                 0);

        return likeTensor;
    }
} // namespace ty
#include "tityos/ty/ops/Addition.h"

namespace ty {
    Tensor add(Tensor &tensor1, Tensor &tensor2) {
        assertSameDevice(tensor1, tensor2);

        switch (tensor1.device().type) {
            case DeviceType::CPU:
            return cpuAdd(tensor1, tensor2);
            case DeviceType::CUDA:
            return cudaAdd(tensor1, tensor2);
        }

        return tensor1;
    }
} // namespace ty
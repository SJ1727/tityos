#include "tityos/ty/tensor/DType.h"

namespace ty {
    size_t dtypeSize(DType type) {
        switch (type) {
        case DType::float16:
            return sizeof(float) / 2;
        case DType::float32:
            return sizeof(float);
        case DType::float64:
            return sizeof(double);
        case DType::int8:
            return sizeof(int8_t);
        case DType::int16:
            return sizeof(int16_t);
        case DType::int32:
            return sizeof(int32_t);
        case DType::int64:
            return sizeof(int64_t);
        case DType::uint8:
            return sizeof(uint8_t);
        case DType::uint16:
            return sizeof(uint16_t);
        case DType::uint32:
            return sizeof(uint32_t);
        case DType::uint64:
            return sizeof(uint64_t);
        case DType::boolean:
            return sizeof(char);

        default:
            return 0;
        }
    }
}; // namespace ty
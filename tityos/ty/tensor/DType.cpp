#include "tityos/ty/tensor/DType.h"

namespace ty {
    size_t dtypeSize(DType type) {
        switch (type) {
        case DType::float16: // No native c++ support for float16. Will deal with this some other day
            return sizeof(uint16_t);
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
            return sizeof(uint8_t);

        default:
            return 0;
        }
    }

    std::string dtypeToString(DType type) {
        switch (type) {
        case DType::float16: // No native c++ support for float16. Will deal with this some other day
            return "Float16";
        case DType::float32:
            return "Float32";
        case DType::float64:
            return "Float64";
        case DType::int8:
            return "Int8";
        case DType::int16:
            return "Int16";
        case DType::int32:
            return "Int32";
        case DType::int64:
            return "Int64";
        case DType::uint8:
            return "UInt8";
        case DType::uint16:
            return "UInt16";
        case DType::uint32:
            return "UInt32";
        case DType::uint64:
            return "UInt64";
        case DType::boolean:
            return "Boolean";

        default:
            return "None";
        }
    }

    bool dtypeSupportsGrad(DType type) {
        return type == DType::float16 || type == DType::float32 || type == DType::float64;
    }
}; // namespace ty
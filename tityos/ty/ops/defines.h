#pragma once

namespace ty {

#define TITYOS_TYPED_FUNC_SWITCH(dtype, errMessage, FUNC, ...)                                     \
    switch (dtype) {                                                                               \
    case DType::float16:                                                                           \
        FUNC(uint16_t, __VA_ARGS__);                                                               \
        break;                                                                                     \
    case DType::float32:                                                                           \
        FUNC(float, __VA_ARGS__);                                                                  \
        break;                                                                                     \
    case DType::float64:                                                                           \
        FUNC(double, __VA_ARGS__);                                                                 \
        break;                                                                                     \
    case DType::int8:                                                                              \
        FUNC(int8_t, __VA_ARGS__);                                                                 \
        break;                                                                                     \
    case DType::int16:                                                                             \
        FUNC(int16_t, __VA_ARGS__);                                                                \
        break;                                                                                     \
    case DType::int32:                                                                             \
        FUNC(int32_t, __VA_ARGS__);                                                                \
        break;                                                                                     \
    case DType::int64:                                                                             \
        FUNC(int64_t, __VA_ARGS__);                                                                \
        break;                                                                                     \
    case DType::uint8:                                                                             \
        FUNC(uint8_t, __VA_ARGS__);                                                                \
        break;                                                                                     \
    case DType::uint16:                                                                            \
        FUNC(uint16_t, __VA_ARGS__);                                                               \
        break;                                                                                     \
    case DType::uint32:                                                                            \
        FUNC(uint32_t, __VA_ARGS__);                                                               \
        break;                                                                                     \
    case DType::uint64:                                                                            \
        FUNC(uint64_t, __VA_ARGS__);                                                               \
        break;                                                                                     \
    default:                                                                                       \
        throw std::runtime_error(errMessage);                                                      \
    }

} // namespace ty
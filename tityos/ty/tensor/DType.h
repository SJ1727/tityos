#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "tityos/ty/common/defines.h"

namespace ty {
    enum class TITYOS_API DType : int {
        float16 = 0,
        float32 = 1,
        float64 = 2,
        int8 = 3,
        int16 = 4,
        int32 = 5,
        int64 = 6,
        uint8 = 7,
        uint16 = 8,
        uint32 = 9,
        uint64 = 10,
        boolean = 11
    };

    size_t dtypeSize(DType type);

    std::string dtypeToString(DType type);

    bool dtypeSupportsGrad(DType type);
};
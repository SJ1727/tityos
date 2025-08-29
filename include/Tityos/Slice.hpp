#pragma once

#include <format>
#include <stdexcept>
#include <climits>

#define OPEN_END INT_MAX

namespace Tityos {
    struct Slice {
        int start;
        int end;
        int stride;
        bool startOpen;
        bool endOpen;

        Slice() : Slice(OPEN_END, OPEN_END) {}
        Slice(int sliceStart, int sliceEnd, int stride);
        Slice(int sliceStart, int sliceEnd) : Slice(sliceStart, sliceEnd, 1) {}
        Slice(int v) : Slice(v, v + 1) {}

        bool operator==(const Slice &other) const;
        bool operator==(int v) const;
        Slice &operator++();
        Slice operator++(int);
    };
} // namespace Tityos
#pragma once

#include <compare>
#include <format>
#include <stdexcept>

namespace Tityos {
    struct Slice {
        int start;
        int end;
        int stride;

        Slice() = default;
        Slice(int start, int end, int stride);
        Slice(int start, int end) : Slice(start, end, 1) {}
        Slice(int v) : Slice(v, v + 1) {}

        std::strong_ordering operator<=>(int v) const;
        bool operator==(const Slice &other) const;
        bool operator==(int v) const;
        Slice &operator++();
        Slice operator++(int);
    };
} // namespace Tityos
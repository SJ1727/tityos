#pragma once

#include <climits>
#include <format>
#include <stdexcept>

#define OPEN_END INT_MAX

namespace ty {
    struct Index {
        int start;
        int end;
        int stride;
        bool startOpen;
        bool endOpen;

        Index() : Index(OPEN_END, OPEN_END) {}
        Index(int sliceStart, int sliceEnd, int stride);
        Index(int sliceStart, int sliceEnd) : Index(sliceStart, sliceEnd, 1) {}
        Index(int v) : Index(v, v + 1) {}

        bool operator==(const Index &other) const;
        bool operator==(int v) const;
        Index &operator++();
        Index operator++(int);
    };
} // namespace ty
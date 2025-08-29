#include "Tityos/Slice.hpp"

namespace Tityos {
    Slice::Slice(int sliceStart, int sliceEnd, int stride) : stride(stride) {
        startOpen = false;
        endOpen = false;

        start = sliceStart;
        end = sliceEnd;

        if (sliceStart == OPEN_END) {
            startOpen = true;
            start = 0;
        }
        if (sliceEnd == OPEN_END) {
            endOpen = true;
            end = 0;
        }
    }

    bool Slice::operator==(const Slice &other) const {
        return other.start == start && other.end == end;
    }

    bool Slice::operator==(int v) const {
        return v == start && v == end;
    }

    Slice &Slice::operator++() {
        start++;
        end++;
        return *this;
    }

    Slice Slice::operator++(int) {
        Slice temp = *this;
        start++;
        end++;
        return temp;
    }
} // namespace Tityos
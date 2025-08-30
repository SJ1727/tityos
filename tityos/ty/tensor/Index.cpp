#include "tityos/ty/tensor/Index.h"

namespace ty {
    Index::Index(int IndexStart, int IndexEnd, int stride) : stride(stride) {
        startOpen = false;
        endOpen = false;

        start = IndexStart;
        end = IndexEnd;

        if (IndexStart == OPEN_END) {
            startOpen = true;
            start = 0;
        }
        if (IndexEnd == OPEN_END) {
            endOpen = true;
            end = 0;
        }
    }

    bool Index::operator==(const Index &other) const {
        return other.start == start && other.end == end;
    }

    bool Index::operator==(int v) const {
        return v == start && v == end;
    }

    Index &Index::operator++() {
        start++;
        end++;
        return *this;
    }

    Index Index::operator++(int) {
        Index temp = *this;
        start++;
        end++;
        return temp;
    }
} // namespace ty
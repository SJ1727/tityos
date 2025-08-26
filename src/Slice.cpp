#include "Tityos/Slice.hpp"

namespace Tityos {
    Slice::Slice(int start, int end, int stride) : start(start), end(end), stride(stride) {
        if (end < start) {
            throw std::invalid_argument(
                std::format("End cannot be less than start in slice. {} < {}", end, start));
        }
    }

    std::strong_ordering Slice::operator<=>(int v) const {
        if (start < v && end < v) {
            return std::strong_ordering::less;
        }
        if (start > v && end > v) {
            return std::strong_ordering::greater;
        }
        return std::strong_ordering::equivalent;
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
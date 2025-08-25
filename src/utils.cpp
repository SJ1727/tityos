#include "utils.hpp"

namespace Tityos {
    int vectorElementProduct(std::vector<int> vec) {
        return std::accumulate(vec.begin(), vec.end(), 1, std::multiplies<int>());
    }
} // namespace Tityos
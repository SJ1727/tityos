#pragma once

#include <algorithm>
#include <compare>
#include <format>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "Tityos/Tensor/Slice.hpp"
#include "utils.hpp"

namespace Tityos {
    namespace Tensor {
        class TensorBase {
          public:
            virtual ~TensorBase() = default;

            virtual std::vector<int> shape() const = 0;
            virtual void print() const = 0;

          private:
            int tensorIndexToFlat(std::vector<int> index) const;
            void printRecurse(int dim, std::vector<int> idx) const;
        };
    } // namespace Tensor
} // namespace Tityos
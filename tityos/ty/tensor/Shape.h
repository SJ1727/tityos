#pragma once

#include <numeric>
#include <stdexcept>
#include <vector>

namespace ty {
    class Shape {
      public:
        explicit Shape(std::vector<int> dims);
        Shape(std::initializer_list<int> dims);
        ~Shape() = default;

        int numDims() const;

        int numElements() const;

        int dim(const size_t dim) const;

        int &operator[](const size_t dim);

        bool operator==(const Shape &other) const;
        bool operator==(const std::initializer_list<int> &other) const;

      private:
        std::vector<int> dims_;
    };
} // namespace ty
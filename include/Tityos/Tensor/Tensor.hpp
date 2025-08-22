#pragma once

#include <vector>
#include <iostream>
#include <stdexcept>
#include <format>
#include <compare>
#include <memory>
#include <algorithm>
#include <numeric>
#include <iomanip>

namespace Tityos
{
    namespace Tensor
    {
        struct Slice
        {
            int start;
            int end;
            int stride;

            Slice() = default;
            Slice(int start, int end, int stride);
            Slice(int start, int end) : Slice(start, end, 1) {}
            Slice(int v) : Slice(v, v + 1) {}

            std::strong_ordering operator<=>(int v) const;
            bool operator==(int v) const;
        };

        class Tensor
        {
        public:
            virtual ~Tensor() = default;

            virtual std::vector<int> shape() const = 0;
            virtual void print() const = 0;
        };

        class FloatTensor : public Tensor
        {
        public:
            FloatTensor(const std::vector<int> &shape);
            FloatTensor(const std::vector<int> &shape, std::vector<float> data);
            FloatTensor(std::shared_ptr<std::vector<float>> data, const std::vector<int> &dataShape, const std::vector<int> &shape, std::vector<int> offsets);
            std::vector<int> shape() const override;
            void print() const override;

            FloatTensor at(std::vector<Slice> slices) const;
            float item() const;

        private:
            int tensorIndexToFlat(std::vector<int> index) const;
            void printRecurse(int dim, std::vector<int> idx) const;

        private:
            std::vector<int> shape_;
            std::vector<int> offsets_;
            std::vector<int> dataShape_;
            std::shared_ptr<std::vector<float>> data_;
        };
    }
}
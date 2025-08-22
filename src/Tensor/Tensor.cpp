#include "Tityos/Tensor/Tensor.hpp"

namespace Tityos
{
    namespace Tensor
    {
        Slice::Slice(int start, int end, int stride) : start(start), end(end), stride(stride)
        {
            if (end < start)
            {
                throw std::invalid_argument(std::format("End cannot be less than start in slice. {} < {}", end, start));
            }
        }

        std::strong_ordering Slice::operator<=>(int v) const
        {
            if (start < v && end < v)
                return std::strong_ordering::less;
            if (start > v && end > v)
                return std::strong_ordering::greater;
            return std::strong_ordering::equivalent;
        }

        bool Slice::operator==(int v) const
        {
            return v == start && v == end;
        }

        FloatTensor::FloatTensor(const std::vector<int> &shape) : shape_(shape), dataShape_(shape)
        {
            int totalSize = 1;

            for (int i = 0; i < shape.size(); i++)
            {
                if (shape[i] < 1)
                {
                    throw std::invalid_argument(std::format("Dims should be positive: Dim at position {} is less than 1. {} < 1", i, shape[i]));
                }
            }

            for (int dim : shape)
                totalSize *= dim;
            data_ = std::make_unique<std::vector<float>>(totalSize, 0.0f);

            offsets_.resize(shape.size());
            offsets_.assign(shape.size(), 1);
        }

        FloatTensor::FloatTensor(const std::vector<int> &shape, std::vector<float> data) : shape_(shape), dataShape_(shape)
        {
            for (int i = 0; i < shape.size(); i++)
            {
                if (shape[i] < 1)
                {
                    throw std::invalid_argument(std::format("Dims should be positive: Dim at position {} is less than 1. {} < 1", i, shape[i]));
                }
            }

            if (data.size() != std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()))
            {
                throw std::invalid_argument("Shape does not fit for size of data");
            }

            data_ = std::make_shared<std::vector<float>>(data);

            offsets_.resize(shape.size());
            offsets_.assign(shape.size(), 1);
        }

        FloatTensor::FloatTensor(std::shared_ptr<std::vector<float>> data, const std::vector<int> &dataShape, const std::vector<int> &shape, std::vector<int> offsets) : data_(data), dataShape_(dataShape), shape_(shape), offsets_(offsets)
        {
            // TODO: Error checking
            for (int i = 0; i < shape.size(); i++)
            {
                if (shape[i] < 1)
                {
                    throw std::invalid_argument(std::format("Dims should be positive: Dim at position {} is less than 1. {} < 1", i, shape[i]));
                }
            }
        }

        std::vector<int> FloatTensor::shape() const
        {
            return shape_;
        }

        void FloatTensor::print() const
        {
            printRecurse(0, {});
        }

        void FloatTensor::printRecurse(int dim, std::vector<int> idx) const
        {
            if (dim == shape_.size())
            {
                int flatIndex = tensorIndexToFlat(idx);
                std::cout << std::setprecision(4) << (*data_)[flatIndex];
                return;
            }

            std::cout << "[";
            for (int i = 0; i < shape_[dim]; ++i)
            {
                std::vector<int> next_idx = idx;
                next_idx.push_back(i);

                printRecurse(dim + 1, next_idx);
                if (i < shape_[dim] - 1)
                {
                    std::cout << (dim == shape_.size() - 1 ? ", " : ",\n ");
                }
            }
            std::cout << "]";
        }

        FloatTensor FloatTensor::at(std::vector<Slice> slices) const
        {
            std::vector<int> shape;
            std::vector<int> offsets;
            Slice currSlice;

            if (slices.size() != shape_.size())
            {
                throw std::runtime_error(std::format("Too many indices provided: Tensor has dim {} but {} indices where provided", shape_.size(), slices.size()));
            }

            for (int i = 0; i < slices.size(); i++)
            {
                if (slices[i] < 0)
                {
                    throw std::runtime_error(std::format("Indices must be positive: Index at position {} is less than 0", i));
                }

                if (slices[i].start >= shape_[i] || slices[i].end > shape_[i])
                {
                    throw std::runtime_error(std::format("Index to larger: Slice is larger than dim size at position {}. Dim:{} Slice:{}-{}", i, shape_[i], slices[i].start, slices[i].end));
                }
            }

            shape.resize(slices.size());
            offsets.resize(slices.size());

            // TODO: Deal with nested slicing
            for (int i = 0; i < slices.size(); i++)
            {
                currSlice = slices[i];

                shape[i] = currSlice.end - currSlice.start;
                offsets[i] = currSlice.start;
            }

            return FloatTensor(data_, shape_, shape, offsets);
        }

        float FloatTensor::item() const
        {
            if (!std::all_of(shape_.begin(), shape_.end(), [](int x)
                             { return x == 1; }))
            {
                throw std::runtime_error("Cannot get item from tensor with more than 1 element");
            }

            std::vector<int> firstElementIndex(shape_.size(), 0);

            return (*data_)[tensorIndexToFlat(firstElementIndex)];
        }

        int FloatTensor::tensorIndexToFlat(std::vector<int> index) const
        {
            int dataIndex = 0;
            int stepSize = 1;

            for (int i = dataShape_.size() - 1; i >= 0; i--)
            {
                dataIndex += (index[i] + offsets_[i]) * stepSize;
                stepSize *= dataShape_[i];
            }

            return dataIndex;
        }
    }
}
#include "Tityos/Tensor.hpp"

namespace Tityos {
    template <typename T>
    Tensor<T>::Tensor(std::vector<T> data, const std::vector<int> &shape) : shape_(shape) {
        for (int i = 0; i < shape.size(); i++) {
            if (shape[i] < 1) {
                throw std::invalid_argument(std::format(
                    "Dims should be positive: Dim at position {} is less than 1. {} < 1", i,
                    shape[i]));
            }
        }

        if (data.size() != vectorElementProduct(shape)) {
            throw std::invalid_argument("Shape does not fit for size of data");
        }

        int stride = 1;
        strides_.resize(shape.size());

        for (int i = shape.size() - 1; i >= 0; i--) {
            strides_[i] = stride;
            stride *= shape[i];
        }

        data_ = std::make_shared<std::vector<T>>(data);
        offset_ = 0;
    }

    template <typename T>
    Tensor<T>::Tensor(std::shared_ptr<std::vector<T>> data, const std::vector<int> &strides_,
                      const std::vector<int> &shape, int offset)
        : data_(data), strides_(strides_), shape_(shape), offset_(offset) {
        // TODO: Error checking
        for (int i = 0; i < shape.size(); i++) {
            if (shape[i] < 1) {
                throw std::invalid_argument(std::format(
                    "Dims should be positive: Dim at position {} is less than 1. {} < 1", i,
                    shape[i]));
            }
        }
    }

    template <typename T> int Tensor<T>::size() const {
        return vectorElementProduct(shape_);
    }

    template <typename T> int Tensor<T>::numDims() const {
        return shape_.size();
    }

    template <typename T> std::vector<int> Tensor<T>::shape() const {
        return shape_;
    }

    template <typename T> void Tensor<T>::print() const {
        printRecurse(0, {});
    }

    template <typename T>
    std::shared_ptr<std::vector<T>> Tensor<T>::data() const {
        return data_;
    }

    template <typename T> Tensor<T> Tensor<T>::at(const std::vector<Slice> &slices) const {
        std::vector<int> shape;
        int start;
        int end;

        if (slices.size() != shape_.size()) {
            throw std::invalid_argument(
                std::format("Number of indices does not match dimensionality: Tensor has "
                            "dimensionality {} but {} "
                            "indices where provided",
                            shape_.size(), slices.size()));
        }

        for (int i = 0; i < slices.size(); i++) {
            // TODO: Allow neagtive indices
            if (slices[i].start < 0 || slices[i].end < 0) {
                throw std::runtime_error(std::format(
                    "Indices must be positive: Index at position {} is less than 0", i));
            }

            if (slices[i].start >= shape_[i] || slices[i].end > shape_[i]) {
                throw std::out_of_range(
                    std::format("Index out of range. Cannot access range {}:{} from dim of size{}",
                                slices[i].start, slices[i].end, shape_[i]));
            }
        }

        shape.resize(slices.size());
        int offset = offset_;

        // Calculating shape and offsets of slice
        for (int i = 0; i < slices.size(); i++) {
            // Handles open ends
            start = slices[i].startOpen ? 0 : slices[i].start;
            end = slices[i].endOpen ? shape_[i] : slices[i].end;

            shape[i] = end - start;
            offset += start * strides_[i];
        }

        return Tensor<T>(data_, strides_, shape, offset);
    }

    template <typename T> T Tensor<T>::item() const {
        if (!std::all_of(shape_.begin(), shape_.end(), [](int x) { return x == 1; })) {
            throw std::runtime_error("Cannot get item from tensor with more than 1 element");
        }

        std::vector<int> firstElementIndex(shape_.size(), 0);

        return (*data_)[getFlatIndex(firstElementIndex)];
    }

    template <typename T> bool Tensor<T>::isContiguous() const {
        bool mismatchFound = false;
        int expectedStride = 1;

        for (int i = shape_.size() - 1; i >= 0; i--) {
            if (expectedStride != strides_[i]) {
                mismatchFound = true;
            }

            if (mismatchFound && shape_[i] != 1) {
                return false;
            }

            expectedStride *= shape_[i];
        }

        return true;
    }

    template <typename T> Tensor<T> Tensor<T>::contiguous() const {
        if (this->isContiguous()) {
            return Tensor<T>(data_, strides_, shape_, offset_);
        } else {
            return this->clone();
        }
    }

    template <typename T> Tensor<T> Tensor<T>::clone() const {
        return Tensor<T>(this->getDataFlat(), shape_);
    }

    template <typename T> void Tensor<T>::permute(const std::vector<int> &dims) {
        std::vector<int> newShape(this->numDims(), 0);
        std::vector<int> newStrides(this->numDims(), 0);

        for (int i = 0; i < dims.size(); i++) {
            newShape[i] = shape_[dims[i]];
            newStrides[i] = strides_[dims[i]];
        }

        shape_ = newShape;
        strides_ = newStrides;
    }

    template <typename T> void Tensor<T>::transpose(int dim1, int dim2) {
        std::vector<int> permuteInput(this->numDims());

        for (int i = 0; i < this->size(); i++) {
            permuteInput[i] = i;
        }

        // Swap the two dims but keep all others the same
        permuteInput[dim1] = dim2;
        permuteInput[dim2] = dim1;

        permute(permuteInput);
    }

    template <typename T> void Tensor<T>::transpose() {
        transpose(this->numDims() - 1, this->numDims() - 2);
    }

    template <typename T> Tensor<T> Tensor<T>::reshape(const std::vector<int> &newShape) {
        return Tensor<T>(this->getDataFlat(), newShape);
    }

    template <typename T> std::vector<T> Tensor<T>::getDataFlat() const {
        std::vector<T> flattendData;
        std::vector<int> index(shape_.size(), 0);
        flattendData.resize(this->size());

        // Loops through the data such that the result is contiguous
        for (int i = 0; i < this->size(); i++) {
            flattendData[i] = (*data_)[getFlatIndex(index)];

            for (int j = shape_.size() - 1; j >= 0; j--) {
                index[j]++;
                if (index[j] != shape_[j]) {
                    break;
                }
                index[j] = 0;
            }
        }

        return flattendData;
    }

    template <typename T> int Tensor<T>::getFlatIndex(const std::vector<int> &index) const {
        int dataIndex = offset_;

        for (int i = 0; i < strides_.size(); i++) {
            dataIndex += index[i] * strides_[i];
        }

        return dataIndex;
    }

    template <typename T> void Tensor<T>::printRecurse(int dim, std::vector<int> idx) const {
        if (dim == shape_.size()) {
            int flatIndex = getFlatIndex(idx);
            std::cout << std::setprecision(4) << (*data_)[flatIndex];
            return;
        }

        std::cout << "[";
        for (int i = 0; i < shape_[dim]; i++) {
            std::vector<int> next_idx = idx;
            next_idx.push_back(i);

            printRecurse(dim + 1, next_idx);
            if (i < shape_[dim] - 1) {
                std::cout << (dim == shape_.size() - 1 ? ", " : ",\n ");
            }
        }
        std::cout << "]";
    }
} // namespace Tityos

template class Tityos::Tensor<float>;
template class Tityos::Tensor<double>;
template class Tityos::Tensor<int>;
template class Tityos::Tensor<bool>;
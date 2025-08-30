#include "tityos/ty/tensor/Tensor.h"

namespace ty {
    template <typename T>
    Tensor<T>::Tensor(std::vector<T> data, const Shape &shape) : shape_(shape) {
        for (int i = 0; i < shape.numDims(); i++) {
            if (shape.dim(i) < 0) {
                throw std::invalid_argument(std::format(
                    "Dims should be non-negative: Dim at position {} is less than 0. {} < 0", i,
                    shape.dim(i)));
            }
        }

        if (data.size() != shape.numElements()) {
            throw std::invalid_argument("Shape does not fit for size of data");
        }

        int stride = 1;
        strides_.resize(shape.numDims());

        for (int i = shape.numDims() - 1; i >= 0; i--) {
            strides_[i] = stride;
            stride *= shape.dim(i);
        }

        data_ = std::make_shared<std::vector<T>>(data);
        offset_ = 0;
    }

    template <typename T>
    Tensor<T>::Tensor(std::shared_ptr<std::vector<T>> data, const std::vector<int> &strides_,
                      const Shape &shape, int offset)
        : data_(data), strides_(strides_), shape_(shape), offset_(offset) {
        // TODO: Error checking
        for (int i = 0; i < shape.numDims(); i++) {
            if (shape.dim(i) < 0) {
                throw std::invalid_argument(std::format(
                    "Dims should be non-negative: Dim at position {} is less than 0. {} < 0", i,
                    shape.dim(i)));
            }
        }
    }

    template <typename T> Shape Tensor<T>::shape() const {
        return shape_;
    }

    template <typename T> void Tensor<T>::print() const {
        printRecurse(0, {});
    }

    template <typename T> std::shared_ptr<std::vector<T>> Tensor<T>::data() const {
        return data_;
    }

    template <typename T> Tensor<T> Tensor<T>::at(const std::vector<Index> &slices) const {
        Shape shape(std::vector(slices.size(), 0));
        int start;
        int end;

        if (slices.size() != shape_.numDims()) {
            throw std::invalid_argument(
                std::format("Number of indices does not match dimensionality: Tensor has "
                            "dimensionality {} but {} "
                            "indices where provided",
                            shape_.numDims(), slices.size()));
        }

        for (int i = 0; i < slices.size(); i++) {
            // TODO: Allow neagtive indices
            if (slices[i].start < 0 || slices[i].end < 0) {
                throw std::runtime_error(std::format(
                    "Indices must be positive: Index at position {} is less than 0", i));
            }

            if (slices[i].start >= shape_.dim(i) || slices[i].end > shape_.dim(i)) {
                throw std::out_of_range(
                    std::format("Index out of range. Cannot access range {}:{} from dim of size{}",
                                slices[i].start, slices[i].end, shape_.dim(i)));
            }
        }

        int offset = offset_;

        // Calculating shape and offsets of slice
        for (int i = 0; i < slices.size(); i++) {
            // Handles open ends
            start = slices[i].startOpen ? 0 : slices[i].start;
            end = slices[i].endOpen ? shape_.dim(i) : slices[i].end;

            shape[i] = end - start;
            offset += start * strides_[i];
        }

        return Tensor<T>(data_, strides_, shape, offset);
    }

    template <typename T> T Tensor<T>::itemAt(const std::vector<int> &index) const {
        return (*data_)[getFlatIndex(index)];
    }

    template <typename T> T Tensor<T>::item() const {
        if (shape_.numElements() != 1) {
            throw std::runtime_error("Cannot get item from tensor with more than 1 element");
        }

        std::vector<int> firstElementIndex(shape_.numDims(), 0);

        return (*data_)[getFlatIndex(firstElementIndex)];
    }

    template <typename T> bool Tensor<T>::isContiguous() const {
        bool mismatchFound = false;
        int expectedStride = 1;

        for (int i = shape_.numDims() - 1; i >= 0; i--) {
            if (expectedStride != strides_[i]) {
                mismatchFound = true;
            }

            if (mismatchFound && shape_.dim(i) != 1) {
                return false;
            }

            expectedStride *= shape_.dim(i);
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
        Shape newShape(std::vector<int>(shape_.numDims(), 0));
        std::vector<int> newStrides(shape_.numDims(), 0);

        for (int i = 0; i < dims.size(); i++) {
            newShape[i] = shape_.dim(dims[i]);
            newStrides[i] = strides_[dims[i]];
        }

        shape_ = newShape;
        strides_ = newStrides;
    }

    template <typename T> void Tensor<T>::transpose(int dim1, int dim2) {
        std::vector<int> permuteInput(shape_.numDims());

        for (int i = 0; i < shape_.numElements(); i++) {
            permuteInput[i] = i;
        }

        // Swap the two dims but keep all others the same
        permuteInput[dim1] = dim2;
        permuteInput[dim2] = dim1;

        permute(permuteInput);
    }

    template <typename T> void Tensor<T>::transpose() {
        transpose(shape_.numDims() - 1, shape_.numDims() - 2);
    }

    template <typename T> Tensor<T> Tensor<T>::reshape(const Shape &newShape) {
        return Tensor<T>(this->getDataFlat(), newShape);
    }

    template <typename T> void Tensor<T>::operator=(const Tensor<T> &other) {
        std::vector<int> index(shape_.numDims(), 0);
        std::vector<int> otherIndex(other.shape().numDims(), 0);
        int shapeDifference = shape_.numDims() - other.shape().numDims();

        // Checking number of dimensions
        if (shapeDifference < 0) {
            throw std::runtime_error(
                std::format("Incorrect number of dimensions: tensor has {} dimensions "
                            "when expected {} at most",
                            other.shape().numDims(), shape_.numDims()));
        }

        // Checking if size of other tensor is suitable for broadcasting
        for (int i = 0; i < other.shape().numDims(); i++) {
            if (other.shape().dim(i) != shape_.dim(i + shapeDifference) && other.shape().dim(i) != 1) {
                throw std::runtime_error(
                    std::format("Shape mismatch at dimension {}: expected {} or 1, but got {}.", i,
                                shape_.dim(i + shapeDifference), other.shape().dim(i)));
            }
        }

        for (int i = 0; i < shape_.numElements(); i++) {
            // Gets the other tensor index, accounting for broadcasting
            for (int j = 0; j < other.shape().numDims(); j++) {
                if (other.shape().dim(j) == 1) {
                    otherIndex[j] = 0;
                } else {
                    otherIndex[j] = index[shapeDifference + j];
                }
            }

            (*data_)[getFlatIndex(index)] = other.itemAt(otherIndex);

            // Goes to the next index
            for (int j = shape_.numDims() - 1; j >= 0; j--) {
                index[j]++;
                if (index[j] != shape_.dim(j)) {
                    break;
                }
                index[j] = 0;
            }
        }
    }

    template <typename T> std::vector<T> Tensor<T>::getDataFlat() const {
        if (data_->size() == shape_.numElements()) {
            return *data_;
        }

        std::vector<T> flattendData;
        std::vector<int> index(shape_.numDims(), 0);
        flattendData.resize(shape_.numElements());

        // Loops through the data such that the result is contiguous
        for (int i = 0; i < shape_.numElements(); i++) {
            flattendData[i] = this->itemAt(index);

            for (int j = shape_.numDims() - 1; j >= 0; j--) {
                index[j]++;
                if (index[j] != shape_.dim(j)) {
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
        if (dim == shape_.numDims()) {
            int flatIndex = getFlatIndex(idx);
            std::cout << std::setprecision(4) << (*data_)[flatIndex];
            return;
        }

        std::cout << "[";
        for (int i = 0; i < shape_.dim(dim); i++) {
            std::vector<int> next_idx = idx;
            next_idx.push_back(i);

            printRecurse(dim + 1, next_idx);
            if (i < shape_.dim(dim) - 1) {
                std::cout << (dim == shape_.numDims() - 1 ? ", " : ",\n ");
            }
        }
        std::cout << "]";
    }
} // namespace ty

template class ty::Tensor<float>;
template class ty::Tensor<double>;
template class ty::Tensor<int>;
template class ty::Tensor<bool>;
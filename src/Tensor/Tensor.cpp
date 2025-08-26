#include "Tityos/Tensor/Tensor.hpp"

namespace Tityos {
    namespace Tensor {
        template <typename T>
        Tensor<T>::Tensor(std::vector<T> data, const std::vector<int> &shape) : shape_(shape) {
            for (int i = 0; i < shape.size(); i++) {
                if (shape[i] < 1) {
                    throw std::invalid_argument(std::format(
                        "Dims should be positive: Dim at position {} is less than 1. {} < 1", i,
                        shape[i]));
                }
            }

            if (data.size() !=
                std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>())) {
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

        template <typename T> void Tensor<T>::printRecurse(int dim, std::vector<int> idx) const {
            if (dim == shape_.size()) {
                int flatIndex = tensorIndexToFlat(idx);
                std::cout << std::setprecision(4) << (*data_)[flatIndex];
                return;
            }

            std::cout << "[";
            for (int i = 0; i < shape_[dim]; ++i) {
                std::vector<int> next_idx = idx;
                next_idx.push_back(i);

                printRecurse(dim + 1, next_idx);
                if (i < shape_[dim] - 1) {
                    std::cout << (dim == shape_.size() - 1 ? ", " : ",\n ");
                }
            }
            std::cout << "]";
        }

        template <typename T> Tensor<T> Tensor<T>::at(std::vector<Slice> slices) const {
            std::vector<int> shape;
            std::vector<int> offsets;
            Slice currSlice;

            if (slices.size() != shape_.size()) {
                throw std::invalid_argument(
                    std::format("Too many indices provided: Tensor has dim {} but {} "
                                "indices where provided",
                                shape_.size(), slices.size()));
            }

            for (int i = 0; i < slices.size(); i++) {
                // TODO: Change this for when open ended ranges
                if (slices[i] < 0) {
                    throw std::runtime_error(std::format(
                        "Indices must be positive: Index at position {} is less than 0", i));
                }

                if (slices[i].start >= shape_[i] || slices[i].end > shape_[i]) {
                    throw std::out_of_range(std::format(
                        "Index out of range. Cannot access range {}:{} from dim of size {}",
                        slices[i].start, slices[i].end, shape_[i]));
                }
            }

            shape.resize(slices.size());
            offsets.resize(slices.size());

            int stepSize = 1;
            int offset = 0;

            // TODO: Deal with nested slicing
            for (int i = slices.size() - 1; i >= 0; i--) {
                currSlice = slices[i];

                shape[i] = currSlice.end - currSlice.start;

                offset += (currSlice.start) * stepSize;
                stepSize *= shape_[i];
            }

            return Tensor<T>(data_, shape_, shape, offset);
        }

        template <typename T> T Tensor<T>::item() const {
            if (!std::all_of(shape_.begin(), shape_.end(), [](int x) { return x == 1; })) {
                throw std::runtime_error("Cannot get item from tensor with more than 1 element");
            }

            std::vector<int> firstElementIndex(shape_.size(), 0);

            return (*data_)[tensorIndexToFlat(firstElementIndex)];
        }

        template <typename T> bool Tensor<T>::isContiguous() const {
            bool dimMustbeOne = false;
            int stride = 1;

            for (int i = shape_.size() - 1; i >= 0; i--) {
                if (stride != strides_[i]) {
                    dimMustbeOne = true;
                }

                if (dimMustbeOne && shape_[i] != 1) {
                    return false;
                }

                stride *= shape_[i];
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
            std::vector<T> clonedData;
            std::vector<int> index(shape_.size(), 0);
            clonedData.resize(this->size());

            // Loops through the data such that the result is contiguous
            for (int i = 0; i < this->size(); i++) {
                clonedData[i] = (*data_)[tensorIndexToFlat(index)];

                for (int j = shape_.size() - 1; j >= 0; j--) {
                    index[j]++;
                    if (index[j] != shape_[j]) {
                        break;
                    }
                    index[j] = 0;
                }
            }

            return Tensor<T>(clonedData, shape_);
        }

        template <typename T> int Tensor<T>::tensorIndexToFlat(std::vector<int> index) const {
            int dataIndex = offset_;

            for (int i = 0; i < strides_.size(); i++) {
                dataIndex += index[i] * strides_[i];
            }

            return dataIndex;
        }
    } // namespace Tensor
} // namespace Tityos

template class Tityos::Tensor::Tensor<double>;
template class Tityos::Tensor::Tensor<float>;
template class Tityos::Tensor::Tensor<int>;
template class Tityos::Tensor::Tensor<bool>;
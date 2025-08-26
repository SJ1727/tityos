#include "Tityos/Tensor/Tensor.hpp"
#include <catch2/catch_all.hpp>

#include <random>

using namespace Tityos;

TEST_CASE("Benchmark cloning data", "[tensor][basic][benchmark]") {
    int totalSize = 32 * 128 * 128;

    std::vector<float> data1(totalSize);

    for (int i = 0; i < totalSize; i++) {
        data1[i] = static_cast<float>(i + 1);
    }

    Tensor::Tensor<float> test1(data1, {32, 128, 128});

    BENCHMARK("Large Tensor Cloning") {
        return test1.clone();
    };
}
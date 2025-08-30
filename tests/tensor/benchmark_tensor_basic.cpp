#include "tityos/ty/tityos.h"
#include <catch2/catch_all.hpp>

#include <random>

using namespace ty;

TEST_CASE("Benchmark acessing data", "[tensor][basic][benchmark]") {
    int totalSize = 8 * 8 * 8;

    std::vector<float> data1(totalSize);

    for (int i = 0; i < totalSize; i++) {
        data1[i] = static_cast<float>(i + 1);
    }

    Tensor<float> test1(data1, {64, 8});
    Tensor<float> test2(data1, {2, 2, 2, 2, 2, 2, 2, 2, 2});

    BENCHMARK("Few Dims Access x10 000") {
        for (int i = 0; i < 10000; i++) {
            test1.at({0, 0});
        }

        return test1.at({0, 0});
    };

    BENCHMARK("Many Dims Access x10 000") {
        for (int i = 0; i < 10000; i++) {
            test2.at({0, 0, 0, 0, 0, 0, 0, 0, 0});
        }

        return test2.at({0, 0, 0, 0, 0, 0, 0, 0, 0});
    };
}

TEST_CASE("Benchmark cloning data", "[tensor][basic][benchmark]") {
    int totalSize = 32 * 128 * 128;

    std::vector<float> data1(totalSize);

    for (int i = 0; i < totalSize; i++) {
        data1[i] = static_cast<float>(i + 1);
    }

    Tensor<float> test1(data1, {32, 128, 128});

    BENCHMARK("Large Tensor Cloning") {
        return test1.clone();
    };
}
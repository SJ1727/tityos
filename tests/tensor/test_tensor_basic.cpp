#include "tityos/ty/tityos.h"
#include <catch2/catch_all.hpp>

#include <random>

using namespace ty;

TEST_CASE("Check tensor shape return", "[tensor][basic]") {
    Tensor<float> t1(Shape{3, 2, 4});
    REQUIRE(t1.shape() == Shape({3, 2, 4}));

    Tensor<float> t2(Shape{5});
    REQUIRE(t2.shape() == Shape({5}));
}

TEST_CASE("Access data from tensor", "[tensor][basic]") {
    int totalSize = 3 * 2 * 4;
    std::vector<float> data1(totalSize);

    for (int i = 0; i < totalSize; i++) {
        data1[i] = static_cast<float>(i + 1);
    }

    Tensor<float> test1(data1, {3, 2, 4});

    // Spot checks
    REQUIRE(test1.at({0, 0, 0}).item() == data1[0]);
    REQUIRE(test1.at({2, 1, 2}).item() == data1[22]);
    REQUIRE(test1.at({2, 1, 3}).item() == data1[23]);

    // Open-end checks
    REQUIRE(test1.at({Index(1, OPEN_END), Index(), Index(OPEN_END, 2)}).shape() ==
            Shape({2, 2, 2}));

    // Loop-based consistency check
    for (int i = 0; i < totalSize; i++) {
        int z = i / (2 * 4);
        int y = (i / 4) % 2;
        int x = i % 4;
        REQUIRE(test1.at({z, y, x}).item() == data1[i]);
    }

    // Nested access
    REQUIRE(test1.at({Index(1, 3), 0, 0}).at({1, 0, 0}).item() == data1[16]);

    // Out-of-bounds access
    REQUIRE_THROWS_AS(test1.at({3, 0, 0}), std::out_of_range);
    REQUIRE_THROWS_AS(test1.at({0, 2, 0}), std::out_of_range);
    REQUIRE_THROWS_AS(test1.at({0}), std::invalid_argument);
}

TEST_CASE("Cloning tensor", "[tensor][basic]") {
    int totalSize = 3 * 2 * 4;

    std::vector<float> data1(totalSize);

    for (int i = 0; i < totalSize; i++) {
        data1[i] = static_cast<float>(i + 1);
    }

    Tensor<float> test1(data1, {3, 2, 4});
    Tensor<float> test1Clone = test1.clone();
    std::vector<int> index(test1.shape().numDims(), 0);

    // Location check
    for (int i = 0; i < test1.shape().numElements(); i++) {
        REQUIRE(test1.itemAt(index) == test1Clone.itemAt(index));

        for (int j = test1.shape().numDims(); j >= 0; j--) {
            index[j]++;
            if (index[j] != test1.shape().dim(j)) {
                break;
            }
            index[j] = 0;
        }
    }

    // Contiguous check
    Tensor<float> test2 = test1.at({Index(1, 3), 1, Index(2, 3)});
    Tensor<float> test2Cloned = test2.clone();

    REQUIRE(test2Cloned.isContiguous());
}

TEST_CASE("Permuting tensor", "[tensor][basic]") {
    int totalSize = 3 * 2 * 4;

    std::vector<float> data1(totalSize);

    for (int i = 0; i < totalSize; i++) {
        data1[i] = static_cast<float>(i + 1);
    }

    Tensor<float> test1Permuted(data1, {3, 2, 4});
    Tensor<float> test1Original(data1, {3, 2, 4});
    test1Permuted.permute({0, 2, 1});

    // Permuted shape
    REQUIRE(test1Permuted.shape() == Shape({3, 4, 2}));

    // Permuted access
    REQUIRE(test1Permuted.itemAt({2, 3, 1}) == test1Original.itemAt({2, 1, 3}));
}

TEST_CASE("Reshape tensor", "[tensor][basic]") {
    int totalSize = 3 * 2 * 4;

    std::vector<float> data1(totalSize);

    for (int i = 0; i < totalSize; i++) {
        data1[i] = static_cast<float>(i + 1);
    }

    Tensor<float> test1Original(data1, {3, 2, 4});
    Tensor<float> test1Reshape = test1Original.reshape({12, 2});

    // Spot check
    REQUIRE(test1Reshape.itemAt({0, 0}) == 1.0f);
    REQUIRE(test1Reshape.itemAt({0, 1}) == 2.0f);
    REQUIRE(test1Reshape.itemAt({5, 0}) == 11.0f);
    REQUIRE(test1Reshape.itemAt({11, 1}) == 24.0f);
}

TEST_CASE("Tensor assignment operator", "[tensor][basic]") {
    int totalSize = 3 * 2 * 4;

    std::vector<float> data1(totalSize);

    for (int i = 0; i < totalSize; i++) {
        data1[i] = static_cast<float>(i + 1);
    }

    Tensor<float> test1(data1, {3, 2, 4});
    Tensor<float> test2({1, 2}, {2, 1});
    Tensor<float> test3({1, 2}, {1, 1, 2, 1});

    test1.at({0, Index(), Index()}) = test2;

    // Check unchaged
    REQUIRE(test1.itemAt({2, 1, 2}) == data1[22]);
    REQUIRE(test1.itemAt({2, 1, 3}) == data1[23]);

    // Check changed
    REQUIRE(test1.itemAt({0, 0, 0}) == 1.0f);
    REQUIRE(test1.itemAt({0, 1, 0}) == 2.0f);

    // Error checking 
    REQUIRE_THROWS_AS(test1.at({1, Index(), Index()}) = test3, std::runtime_error);
}
#include "Tityos/Tensor/Tensor.hpp"
#include <catch2/catch_all.hpp>

#include <random>

using namespace Tityos;

TEST_CASE("Check tensor shape return", "[tensor][basic]") {
    Tensor::FloatTensor t1({3, 2, 4});
    REQUIRE(t1.shape() == std::vector{3, 2, 4});

    Tensor::FloatTensor t2({5});
    REQUIRE(t2.shape() == std::vector{5});

    // Empty tensor creation
    Tensor::FloatTensor t3({});
    REQUIRE(t3.shape() == std::vector<int>{});
}

TEST_CASE("Slice struct comparison", "[slice][basic]") {
    Tensor::Slice s1 = 1;
    Tensor::Slice s2(4, 5);

    REQUIRE(s1 < 3);
    REQUIRE_FALSE(s1 < 0);

    REQUIRE(s2 < 6);
    REQUIRE_FALSE(s2 < 5);

    REQUIRE(s1 == Tensor::Slice(1));
    REQUIRE(s1 != s2);

    REQUIRE_THROWS_AS(Tensor::Slice(5, 4), std::invalid_argument);
}

TEST_CASE("Access data from tensor", "[tensor][basic]") {
    int totalSize = 3 * 2 * 4;
    std::vector<float> data1(totalSize);

    for (int i = 0; i < totalSize; i++) {
        data1[i] = static_cast<float>(i + 1);
    }

    Tensor::FloatTensor test1(data1, {3, 2, 4});

    // Spot checks
    REQUIRE(test1.at({0, 0, 0}).item() == data1[0]);
    REQUIRE(test1.at({2, 1, 2}).item() == data1[22]);
    REQUIRE(test1.at({2, 1, 3}).item() == data1[23]);

    // Loop-based consistency check
    for (int i = 0; i < totalSize; i++) {
        int z = i / (2 * 4);
        int y = (i / 4) % 2;
        int x = i % 4;
        REQUIRE(test1.at({z, y, x}).item() == data1[i]);
    }

    // Out-of-bounds access
    REQUIRE_THROWS_AS(test1.at({3, 0, 0}), std::out_of_range);
    REQUIRE_THROWS_AS(test1.at({0, 2, 0}), std::out_of_range);
    REQUIRE_THROWS_AS(test1.at({0}), std::invalid_argument);
}

TEST_CASE("Cloning data", "[tensor][basic][benchmark]") {
    int totalSize = 3 * 2 * 4;

    std::vector<float> data1(totalSize);

    for (int i = 0; i < totalSize; i++) {
        data1[i] = static_cast<float>(i + 1);
    }

    Tensor::FloatTensor test1(data1, {3, 2, 4});
    Tensor::FloatTensor test1Clone = test1.clone();
    std::vector<Tensor::Slice> index(test1.numDims(), 0);

    // Location check
    for (int i = 0; i < test1.size(); i++) {
        REQUIRE(test1.at(index).item() == test1Clone.at(index).item());

        for (int j = test1.numDims(); j >= 0; j--) {
            index[j]++;
            if (index[j] != test1.shape()[j]) {
                break;
            }
            index[j] = 0;
        }
    }

    // Contiguous check
    Tensor::FloatTensor test2 = test1.at({Tensor::Slice(1, 3), 1, Tensor::Slice(2, 3)});
    Tensor::FloatTensor test2Cloned = test2.clone();

    REQUIRE(test2Cloned.isContiguous());
}
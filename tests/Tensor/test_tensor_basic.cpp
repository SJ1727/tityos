#include <catch2/catch_test_macros.hpp>
#include "Tityos/Tensor/Tensor.hpp"

#include <random>

using namespace Tityos;

TEST_CASE("Check tensor shape return", "[tensor][basic]")
{
    std::vector expected = {3, 2, 4};
    Tensor::FloatTensor test({3, 2, 4});

    REQUIRE(test.shape() == expected);
}

TEST_CASE("Slice struct comparison", "[slice][basic]")
{
    Tensor::Slice test1 = 1;
    Tensor::Slice test2(4, 5);

    REQUIRE(test1 < 3);
    REQUIRE(test2 < 6);
    REQUIRE_FALSE(test2 < 5);
    REQUIRE_THROWS_AS(Tensor::Slice(5, 4), std::invalid_argument);
}

TEST_CASE("Access data from tensor", "[tensor][basic]")
{
    int totalSize = 3 * 3 * 3;

    std::vector<float> data(totalSize);

    for (int i = 0; i < totalSize; ++i) {
        data[i] = static_cast<float>(i + 1);
    }

    Tensor::FloatTensor test({3, 3, 3}, data);

    REQUIRE(test.at({0, 0, 0}).item() == data[0]);
    REQUIRE(test.at({1, 1, 1}).item() == data[13]);
}
#include <catch2/catch_all.hpp>

#include "tityos/ty/tityos.h"

TEST_CASE("Tensor relu", "[ops][activations]") {
    // RELU with float32
    std::vector<float> testData1(48);

    for (int i = 0; i < 48; i++) {
        testData1[i] = static_cast<float>(i - 24);
    }

    ty::Tensor test1(testData1, ty::Shape({1, 8, 6}));

    REQUIRE_NOTHROW([&]() { ty::Tensor result = ty::relu(test1); });

    ty::Tensor result1 = ty::relu(test1);

    for (size_t i = 0; i < 48; i++) {
        REQUIRE(*static_cast<float *>(result1.get(i)) == (testData1[i] > 0 ? testData1[i] : 0));
    }

    // RELU with int64
    std::vector<int64_t> testData2(48);
    for (int i = 0; i < 48; i++) {
        testData2[i] = static_cast<int64_t>(i - 24);
    }

    ty::Tensor test2(testData2, ty::Shape({1, 8, 6}), ty::DType::int64);

    REQUIRE_NOTHROW([&]() { ty::Tensor result = ty::relu(test2); });

    ty::Tensor result2 = ty::relu(test2);

    for (size_t i = 0; i < 48; i++) {
        REQUIRE(*static_cast<int64_t *>(result2.get(i)) == (testData2[i] > 0 ? testData2[i] : 0));
    }
}
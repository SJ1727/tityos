#include <catch2/catch_all.hpp>

#include "tityos/ty/tityos.h"

TEST_CASE("Tensor creation", "[tensor][base]") {
    std::vector<float> testData1(48, 0.0f);
    std::vector<int64_t> testData2(48, 0);

    REQUIRE_NOTHROW([&]() { ty::Tensor test(testData1, ty::Shape({8, 6})); });
    REQUIRE_NOTHROW([&]() { ty::Tensor test(testData1, ty::Shape({1, 1, 8, 6})); });
    REQUIRE_NOTHROW([&]() { ty::Tensor test(testData1, ty::Shape({2, 1, 4, 6})); });

    REQUIRE_NOTHROW([&]() { ty::Tensor test(testData2, ty::Shape({8, 6}), ty::DType::int64); });

    ty::Tensor test(testData1, ty::Shape({1, 8, 6}));

    REQUIRE(test.numElements() == 48);
}

TEST_CASE("Shape Broadcasting", "[shape][base]") {
    ty::Shape shape1({1, 1, 8, 6});
    ty::Shape shape2({2, 2, 1, 8, 1});

    std::vector<int64_t> expected = {2, 2, 1, 8, 6};

    ty::Shape shape3 = ty::broadcastCombineShapes(shape1, shape2);

    for (int i = 0; i < shape3.numDims(); i++) {
        REQUIRE(shape3[i] == expected[i]);
    }

    ty::Shape shape4({4, 1, 8, 6});
    ty::Shape shape5({2, 2, 1, 8, 6});

    REQUIRE_THROWS(ty::broadcastCombineShapes(shape4, shape5));
    REQUIRE_NOTHROW(ty::broadcastCombineShapes(shape2, shape5));
}
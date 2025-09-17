#include <catch2/catch_all.hpp>

#include "tityos/ty/tityos.h"

TEST_CASE("Tensor addition", "[ops][binary]") {
    // Addition without broadcasting
    std::vector<float> testData1(48, 2.0f);
    std::vector<float> testData2(48, 3.0f);

    ty::Tensor test1(testData1, ty::Shape({8, 6}));
    ty::Tensor test2(testData2, ty::Shape({8, 6}));

    REQUIRE_NOTHROW([&]() { ty::Tensor result = ty::add(test1, test2); });

    ty::Tensor result1 = ty::add(test1, test2);

    for (size_t i = 0; i < 48; i++) {
        REQUIRE(*static_cast<float *>(result1.get(i)) == 5.0f);
    }

    // Addition with broadcasting
    std::vector<float> testData3(48, 2.0f);
    std::vector<float> testData4(6, 3.0f);

    ty::Tensor test3(testData3, ty::Shape({8, 6}));
    ty::Tensor test4(testData4, ty::Shape({1, 6}));

    REQUIRE_NOTHROW([&]() { ty::Tensor result = ty::add(test3, test4); });

    ty::Tensor result2 = ty::add(test3, test4);

    for (size_t i = 0; i < 48; i++) {
        REQUIRE(*static_cast<float *>(result2.get(i)) == 5.0f);
    }
}
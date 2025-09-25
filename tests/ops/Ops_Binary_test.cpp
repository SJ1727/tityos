#include <catch2/catch_all.hpp>

#include "tittyos/ty/tittyos.h"

TEST_CASE("Tensor addition", "[ops][binary]") {
    // Addition with float32
    std::vector<float> testData1(48, 2.0f);
    std::vector<float> testData2(6, 3.0f);

    ty::Tensor test1(testData1, ty::Shape({1, 8, 6}));
    ty::Tensor test2(testData2, ty::Shape({1, 1, 6}));

    REQUIRE_NOTHROW([&]() { ty::Tensor result = ty::add(test1, test2); });

    ty::Tensor result1 = ty::add(test1, test2);

    for (size_t i = 0; i < 48; i++) {
        REQUIRE(*static_cast<float *>(result1.get(i)) == 5.0f);
    }

    // Addition with int64
    std::vector<int64_t> testDataInt1(48, 2);
    std::vector<int64_t> testDataInt2(6, 3);

    ty::Tensor testInt1(testDataInt1, ty::Shape({1, 8, 6}), ty::DType::int64);
    ty::Tensor testInt2(testDataInt2, ty::Shape({1, 1, 6}), ty::DType::int64);

    REQUIRE_NOTHROW([&]() { ty::Tensor result = ty::add(testInt1, testInt2); });

    ty::Tensor resultInt1 = ty::add(testInt1, testInt2);

    for (size_t i = 0; i < 48; i++) {
        REQUIRE(*static_cast<int64_t *>(resultInt1.get(i)) == 5);
    }
}

TEST_CASE("Tensor subtraction", "[ops][binary]") {
    // Subtraction with float32
    std::vector<float> testData1(48, 5.0f);
    std::vector<float> testData2(6, 3.0f);

    ty::Tensor test1(testData1, ty::Shape({1, 8, 6}));
    ty::Tensor test2(testData2, ty::Shape({1, 1, 6}));

    REQUIRE_NOTHROW([&]() { ty::Tensor result = ty::subtract(test1, test2); });

    ty::Tensor result1 = ty::subtract(test1, test2);

    for (size_t i = 0; i < 48; i++) {
        REQUIRE(*static_cast<float *>(result1.get(i)) == 2.0f);
    }

    // Subtraction with int64
    std::vector<int64_t> testDataInt1(48, 5);
    std::vector<int64_t> testDataInt2(6, 3);

    ty::Tensor testInt1(testDataInt1, ty::Shape({1, 8, 6}), ty::DType::int64);
    ty::Tensor testInt2(testDataInt2, ty::Shape({1, 1, 6}), ty::DType::int64);

    REQUIRE_NOTHROW([&]() { ty::Tensor result = ty::subtract(testInt1, testInt2); });

    ty::Tensor resultInt1 = ty::subtract(testInt1, testInt2);

    for (size_t i = 0; i < 48; i++) {
        REQUIRE(*static_cast<int64_t *>(resultInt1.get(i)) == 2);
    }
}

TEST_CASE("Tensor multiplication", "[ops][binary]") {
    // Multiplication with float32
    std::vector<float> testData1(48, 2.0f);
    std::vector<float> testData2(6, 3.0f);

    ty::Tensor test1(testData1, ty::Shape({1, 8, 6}));
    ty::Tensor test2(testData2, ty::Shape({1, 1, 6}));

    REQUIRE_NOTHROW([&]() { ty::Tensor result = ty::multiply(test1, test2); });

    ty::Tensor result1 = ty::multiply(test1, test2);

    for (size_t i = 0; i < 48; i++) {
        REQUIRE(*static_cast<float *>(result1.get(i)) == 6.0f); // 2 * 3
    }

    // Multiplication with int64
    std::vector<int64_t> testDataInt1(48, 2);
    std::vector<int64_t> testDataInt2(6, 3);

    ty::Tensor testInt1(testDataInt1, ty::Shape({1, 8, 6}), ty::DType::int64);
    ty::Tensor testInt2(testDataInt2, ty::Shape({1, 1, 6}), ty::DType::int64);

    REQUIRE_NOTHROW([&]() { ty::Tensor result = ty::multiply(testInt1, testInt2); });

    ty::Tensor resultInt1 = ty::multiply(testInt1, testInt2);

    for (size_t i = 0; i < 48; i++) {
        REQUIRE(*static_cast<int64_t *>(resultInt1.get(i)) == 6); // 2 * 3
    }
}

TEST_CASE("Tensor division", "[ops][binary]") {
    // Division with float32
    std::vector<float> testData1(48, 6.0f);
    std::vector<float> testData2(6, 3.0f);

    ty::Tensor test1(testData1, ty::Shape({1, 8, 6}));
    ty::Tensor test2(testData2, ty::Shape({1, 1, 6}));

    REQUIRE_NOTHROW([&]() { ty::Tensor result = ty::divide(test1, test2); });

    ty::Tensor result1 = ty::divide(test1, test2);

    for (size_t i = 0; i < 48; i++) {
        REQUIRE(*static_cast<float *>(result1.get(i)) == 2.0f); // 6 / 3
    }

    // Division with int64
    std::vector<int64_t> testDataInt1(48, 6);
    std::vector<int64_t> testDataInt2(6, 3);

    ty::Tensor testInt1(testDataInt1, ty::Shape({1, 8, 6}), ty::DType::int64);
    ty::Tensor testInt2(testDataInt2, ty::Shape({1, 1, 6}), ty::DType::int64);

    REQUIRE_NOTHROW([&]() { ty::Tensor result = ty::divide(testInt1, testInt2); });

    ty::Tensor resultInt1 = ty::divide(testInt1, testInt2);

    for (size_t i = 0; i < 48; i++) {
        REQUIRE(*static_cast<int64_t *>(resultInt1.get(i)) == 2); // 6 / 3
    }
}

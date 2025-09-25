#include <catch2/catch_all.hpp>

#include "tityos/ty/tityos.h"

TEST_CASE("Tensor multiplication gradient", "[tensor][autograd]") {
    std::vector<float> testData1(48, 2.0f);
    std::vector<float> testData2(48, 3.0f);
    std::vector<float> initialGrad(48, 1.0f);

    ty::Tensor test1(testData1, ty::Shape({1, 8, 6}), ty::DType::float32, ty::DeviceType::CPU, true);
    ty::Tensor test2(testData2, ty::Shape({1, 8, 6}), ty::DType::float32, ty::DeviceType::CPU, true);

    auto resultGrad = std::make_shared<ty::Tensor>(initialGrad, ty::Shape({1, 8, 6}));

    ty::Tensor result = ty::multiply(test1, test2);
    result.backward(resultGrad);

    for (size_t i = 0; i < 48; i++) {
        REQUIRE(*static_cast<float *>(test1.grad()->get(i)) == 3.0f);
        REQUIRE(*static_cast<float *>(test2.grad()->get(i)) == 2.0f);
    }
}
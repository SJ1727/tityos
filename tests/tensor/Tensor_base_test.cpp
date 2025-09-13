#include <catch2/catch_all.hpp>

#include "tityos/ty/tensor/tensor.h"

TEST_CASE("Tensor creation", "[tensor][base]") {
    std::vector<float> testData(48, 0.0f);

    ty::Tensor test1(testData, ty::Shape({8, 6}));
}
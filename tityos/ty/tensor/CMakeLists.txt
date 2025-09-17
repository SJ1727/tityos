cmake_minimum_required(VERSION 3.15)

target_sources(
    tityos
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/Tensor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/TensorIterator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/DType.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Shape.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Slice.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Device.cpp
)

include(${CMAKE_CURRENT_LIST_DIR}/backend/CMakeLists.txt)
cmake_minimum_required(VERSION 3.15)

target_sources(
    tityos
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/Tensor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Index.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Shape.cpp
)
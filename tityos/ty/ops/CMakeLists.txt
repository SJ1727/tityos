cmake_minimum_required(VERSION 3.15)

set(CPU_DIR ${CMAKE_CURRENT_LIST_DIR}/cpu)
set(CUDA_DIR ${CMAKE_CURRENT_LIST_DIR}/cuda)

target_sources(
    tityos
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/BinaryOps.cpp
    ${CPU_DIR}/BinaryOps.cpp
    ${CUDA_DIR}/BinaryOps.cpp

    ${CMAKE_CURRENT_LIST_DIR}/Activations.cpp
    ${CPU_DIR}/Activations.cpp
    ${CUDA_DIR}/Activations.cpp
)
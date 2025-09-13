cmake_minimum_required(VERSION 3.15)

set(CPU_DIR ${CMAKE_CURRENT_LIST_DIR}/cpu)
set(CUDA_DIR ${CMAKE_CURRENT_LIST_DIR}/cuda)

target_sources(
    tityos
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/Addition.cpp
    ${CPU_DIR}/Addition.cpp
    ${CUDA_DIR}/Addition.cpp
)
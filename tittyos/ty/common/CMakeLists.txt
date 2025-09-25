cmake_minimum_required(VERSION 3.15)

target_sources(
    tittyos
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/assert.cpp
)

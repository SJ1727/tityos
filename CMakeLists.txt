cmake_minimum_required(VERSION 3.15)
project(tityos LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_SHARED_LIBS "Build as a shared library" OFF)

set(TY_ROOT_DIR ${PROJECT_SOURCE_DIR}/tityos)

add_library(tityos)

target_include_directories(
    tityos
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)

include(${TY_ROOT_DIR}/ty/CMakeLists.txt)

if(BUILD_SHARED_LIBS)
    target_compile_definitions(tityos PRIVATE TITYOS_EXPORTS)
endif()

add_subdirectory(tests)
cmake_minimum_required(VERSION 3.15)
project(tittyos LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_SHARED_LIBS "Build as a shared library" OFF)

set(TY_ROOT_DIR ${PROJECT_SOURCE_DIR}/tittyos)

add_library(tittyos)

target_include_directories(
    tittyos
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)

include(${TY_ROOT_DIR}/ty/CMakeLists.txt)


if(BUILD_SHARED_LIBS)
    target_compile_definitions(tittyos PRIVATE tittyos_EXPORTS)
endif()

add_subdirectory(tests)
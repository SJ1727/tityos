cmake_minimum_required(VERSION 3.15)
project(tityos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp")

add_library(tityos STATIC
    ${SOURCES}
)

target_include_directories(tityos PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

include(CTest)

add_subdirectory(tests)
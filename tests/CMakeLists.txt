find_package(Catch2 3 REQUIRED)

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "*.cpp")

add_executable(tityos_tests ${TEST_SOURCES})

target_link_libraries(tityos_tests PRIVATE Catch2::Catch2WithMain tityos)

target_include_directories(tityos_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

include(Catch)
catch_discover_tests(tityos_tests)
find_package(Catch2 3 REQUIRED)

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "*.cpp")

add_executable(tittyos_tests ${TEST_SOURCES})

target_link_libraries(tittyos_tests PRIVATE Catch2::Catch2WithMain tittyos)

target_include_directories(tittyos_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

include(Catch)
catch_discover_tests(tittyos_tests)
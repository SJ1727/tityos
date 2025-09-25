cmake_minimum_required(VERSION 3.15)

# --- Library Components ---

include(${CMAKE_CURRENT_LIST_DIR}/tensor/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/ops/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/nn/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/utils/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/common/CMakeLists.txt)
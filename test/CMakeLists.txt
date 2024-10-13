find_package(GTest REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${CMAKE_BINARY_DIR}/external/include)

link_directories(${CMAKE_BINARY_DIR}/external/lib)

include(${CMAKE_SOURCE_DIR}/cmake/options.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/models.cmake)

add_subdirectory(node)
if(${BUILD_PYTHON})
    add_subdirectory(python)
endif()
add_subdirectory(tensor)
add_subdirectory(utils)

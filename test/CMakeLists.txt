find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${CMAKE_BINARY_DIR}/external/include)

link_directories(${CMAKE_BINARY_DIR}/external/lib)

include(${CMAKE_SOURCE_DIR}/cmake/options.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/models.cmake)

if(${BUILD_PYTHON} STREQUAL "ON")
    add_subdirectory(python)
endif()
add_subdirectory(tensor)
add_subdirectory(utils)

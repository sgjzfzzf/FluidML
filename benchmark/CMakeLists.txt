include(${CMAKE_SOURCE_DIR}/cmake/options.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/models.cmake)

include_directories(${CMAKE_BINARY_DIR}/external/include)
include_directories(${CMAKE_BINARY_DIR}/external/include)

link_directories(${CMAKE_BINARY_DIR}/external/lib)

add_executable(bert bert.cpp)
add_dependencies(bert download_bert)
target_link_libraries(bert ${PROJECT_NAME}Static benchmark glog)

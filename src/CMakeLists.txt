include(${CMAKE_SOURCE_DIR}/cmake/options.cmake)

include_directories(${CMAKE_BINARY_DIR}/external/include)

link_directories(${CMAKE_BINARY_DIR}/external/lib)

file(GLOB_RECURSE SRCS "evaluation/*.cpp" "optimization/*.cpp" "structure/*.cpp" "utils/*.cpp" "worker/*.cpp")
file(GLOB_RECURSE PYTHON_BINDS "python/*.cpp")

add_library(LIB_STATIC ${SRCS})
target_link_libraries(LIB_STATIC fmt onnx LLVM MLIR)
if(${BUILD_PYTHON})
    target_link_libraries(LIB_STATIC ${Python3_LIBRARIES})
endif()
set_target_properties(LIB_STATIC PROPERTIES OUTPUT_NAME ${PROJECT_NAME})

if (${BUILD_PYTHON} STREQUAL "ON")
    add_library(LIB_SHARED SHARED ${PYTHON_BINDS})
    target_link_libraries(LIB_SHARED LIB_STATIC ${Python3_LIBRARIES})
else()
    add_library(LIB_SHARED SHARED)
    target_link_libraries(LIB_SHARED LIB_STATIC)
endif()
set_target_properties(LIB_SHARED PROPERTIES OUTPUT_NAME ${PROJECT_NAME})

add_dependencies(LIB_STATIC libfmt libglog libjson)
add_dependencies(LIB_SHARED libfmt libglog libjson)

install(
    TARGETS LIB_STATIC LIB_SHARED
    DESTINATION lib
)

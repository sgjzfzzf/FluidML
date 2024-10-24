include(${CMAKE_SOURCE_DIR}/cmake/options.cmake)

include_directories(${CMAKE_BINARY_DIR}/external/include)

link_directories(${CMAKE_BINARY_DIR}/external/lib)

file(GLOB_RECURSE SRCS "evaluation/*.cpp" "optimization/*.cpp" "structure/*.cpp" "utils/*.cpp" "worker/*.cpp")
file(GLOB_RECURSE PYTHON_BINDS "python/*.cpp")

add_library(LibStatic ${SRCS})
add_dependencies(LibStatic libfmt libglog libjson)
target_link_libraries(LibStatic fmt onnx LLVM MLIR)
if(${BUILD_PYTHON} STREQUAL "ON")
    target_link_libraries(LibStatic ${Python3_LIBRARIES})
endif()
set_target_properties(LibStatic PROPERTIES OUTPUT_NAME ${PROJECT_NAME})

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/empty.cpp "")
add_library(LibShared SHARED ${CMAKE_CURRENT_BINARY_DIR}/empty.cpp)
add_dependencies(LibShared libfmt libglog libjson)
target_link_libraries(LibShared LibStatic)
set_target_properties(LibShared PROPERTIES OUTPUT_NAME ${PROJECT_NAME})

if(${BUILD_PYTHON} STREQUAL "ON")
    add_library(PythonSo SHARED ${PYTHON_BINDS})
    target_link_libraries(PythonSo LibShared)
    set_target_properties(PythonSo PROPERTIES
        OUTPUT_NAME ${PYTHON_PROJECT_NAME}
        PREFIX ""
        SUFFIX ".so")
endif()

install(
    TARGETS LibStatic LibShared
    DESTINATION lib
)

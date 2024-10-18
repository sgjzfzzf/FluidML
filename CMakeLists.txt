cmake_minimum_required(VERSION 3.17)

project(CpuTransformers)

set(CMAKE_CXX_STANDARD_REQUIRED 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(ExternalProject)

option(CMAKE_BUILD_TYPE "Build type" Debug)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_PYTHON "Build python bindings" OFF)
option(DP_DEBUG "Skip the Dynamic Programming optimization for debug" OFF)
option(USE_LOGS "Use logs" OFF)

add_compile_options(-fPIC -Werror)

if(${BUILD_TESTS})
    ExternalProject_Add(
        libbenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.0
        CMAKE_ARGS -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/external 
        BUILD_COMMAND ${CMAKE_COMMAND} --build . --target all
        INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
    )
endif()

ExternalProject_Add(
    libfmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 10.2.1
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/external -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_BUILD_TYPE=Release
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --target all
    INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
)

ExternalProject_Add(
    libglog
    GIT_REPOSITORY https://github.com/google/glog.git
    GIT_TAG v0.7.1
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/external -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_BUILD_TYPE=Release
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --target all
    INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
)

ExternalProject_Add(
    libjson
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/external -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_BUILD_TYPE=Release
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --target all
    INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
)

find_package(LLVM 18.1.6 REQUIRED CONFIG)

include_directories(include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/third_party/onnx)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx)
include_directories(${LLVM_INCLUDE_DIRS})

link_directories(${CMAKE_CURRENT_BINARY_DIR}/third_party/onnx)
link_directories(${LLVM_LIBRARY_DIRS})

add_subdirectory(third_party)
add_subdirectory(src)

if(${BUILD_TESTS})
    enable_testing()
    add_subdirectory(benchmark)
    add_subdirectory(test)
endif()

find_program(CLANG_FORMAT clang-format REQUIRED)
find_program(BLACK_FORMAT black REQUIRED)
add_custom_target(
    format
    COMMAND find ${CMAKE_CURRENT_SOURCE_DIR}/test -name "\"*.py\"" | xargs ${BLACK_FORMAT}
    COMMAND find ${CMAKE_CURRENT_SOURCE_DIR}/benchmark ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/test -name "\"*.cpp\"" -o -name "\"*.h\"" -o -name "\"*.hpp\"" | xargs ${CLANG_FORMAT} -i
)

install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION include/cpu_transformers
)

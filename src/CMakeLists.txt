include_directories(${CMAKE_BINARY_DIR}/external/include)

link_directories(${CMAKE_BINARY_DIR}/external/lib)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0 -g)
    add_compile_definitions(DEBUG)
    add_link_options(-fsanitize=address,undefined)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
else()
    message(FATAL_ERROR "Unknown build type: ${CMAKE_BUILD_TYPE}. Please use Debug or Release.")
endif()

if(BUILD_PYTHON)
    find_package(Python3 3.10 COMPONENTS Development Interpreter REQUIRED)
    include_directories(${Python3_INCLUDE_DIRS})
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import pybind11\nprint(pybind11.get_include())" 
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR)
    include_directories(${PYBIND11_INCLUDE_DIR})
    add_compile_definitions(BUILD_PYTHON)
endif()

if(DP_DEBUG STREQUAL "ON")
    add_compile_definitions(DP_DEBUG)
endif()

file(GLOB_RECURSE SRCS "evaluation/*.cpp" "exception/*.cpp" "optimization/*.cpp" "structure/*.cpp" "utils/*.cpp" "worker/*.cpp")
file(GLOB_RECURSE PYTHON_BINDS "python/*.cpp")

add_library(${PROJECT_NAME}Static ${SRCS})
target_link_libraries(${PROJECT_NAME}Static fmt onnx LLVM MLIR ${Python3_LIBRARIES})
set_target_properties(${PROJECT_NAME}Static PROPERTIES OUTPUT_NAME ${PROJECT_NAME})

if (${BUILD_PYTHON})
    add_library(${PROJECT_NAME}Shared SHARED ${PYTHON_BINDS})
    target_link_libraries(${PROJECT_NAME}Shared ${Python3_LIBRARIES} ${PROJECT_NAME}Static)
    set_target_properties(${PROJECT_NAME}Shared PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
endif()

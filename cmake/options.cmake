add_compile_options(-fPIC)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0 -g)
    add_compile_definitions(DEBUG)
    add_link_options(-fsanitize=address,undefined)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
else()
    message(FATAL_ERROR "Unknown build type: ${CMAKE_BUILD_TYPE}. Please use Debug or Release.")
endif()

if(${BUILD_PYTHON} STREQUAL "ON")
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

if (USE_LOGS STREQUAL "ON")
    add_compile_definitions(USE_LOGS)
endif()

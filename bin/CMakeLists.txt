include(${CMAKE_SOURCE_DIR}/cmake/options.cmake)

find_package(Boost REQUIRED COMPONENTS program_options)

include_directories(${CMAKE_BINARY_DIR}/external/include)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${LibArchive_INCLUDE_DIRS})

link_directories(${CMAKE_BINARY_DIR}/external/lib)

add_executable(compiler compiler.cpp)
target_link_libraries(compiler fmt LIB_STATIC ${Boost_LIBRARIES} ${LibArchive_LIBRARIES})
install(TARGETS compiler DESTINATION bin)

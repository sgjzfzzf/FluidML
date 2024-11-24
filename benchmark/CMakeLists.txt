include(${CMAKE_SOURCE_DIR}/cmake/options.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/models.cmake)

include_directories(${CMAKE_BINARY_DIR}/external/include)

link_directories(${CMAKE_BINARY_DIR}/external/lib)

set(Models "bert" "convbert" "efficientnet" "gptneox" "ibert" "vgg")

foreach(Model IN LISTS Models)
    add_executable(${Model} ${Model}.cpp)
    add_dependencies(${Model} prepare_${Model})
    target_link_libraries(${Model} LibStatic benchmark glog)
    install(TARGETS ${Model} DESTINATION bin)
endforeach()

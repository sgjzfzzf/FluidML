file(GLOB_RECURSE SRCS "exception/*.cpp" "optimization/*.cpp" "structure/*.cpp" "utils/*.cpp" "worker/*.cpp")
file(GLOB_RECURSE PYTHON_BINDS "python/*.cpp")

add_library(${PROJECT_NAME}Static ${SRCS})
target_link_libraries(${PROJECT_NAME}Static fmt onnx LLVM MLIR ${Python3_LIBRARIES})
target_link_directories(${PROJECT_NAME}Static PUBLIC /usr/local/lib/)
set_target_properties(${PROJECT_NAME}Static PROPERTIES OUTPUT_NAME ${PROJECT_NAME})

if (${BUILD_PYTHON})
    add_library(${PROJECT_NAME}Shared SHARED ${PYTHON_BINDS})
    target_link_libraries(${PROJECT_NAME}Shared ${Python3_LIBRARIES} ${PROJECT_NAME}Static)
    set_target_properties(${PROJECT_NAME}Shared PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
endif()

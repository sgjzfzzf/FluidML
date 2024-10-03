find_program(ONNXSIM NAMES onnxsim REQUIRED)

find_package(GTest REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(Wget REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${CMAKE_BINARY_DIR}/external/include)

link_directories(${CMAKE_BINARY_DIR}/external/lib)

# if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/bert_Opset18.onnx)
#     execute_process(
#         COMMAND ${WGET_EXECUTABLE} -q -O bert_Opset18.onnx "https://github.com/onnx/models/raw/main/Natural_Language_Processing/bert_Opset18_transformers/bert_Opset18.onnx"
#         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#     )
#     execute_process(
#         COMMAND ${ONNXSIM} bert_Opset18.onnx bert_Opset18.onnx
#         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#     )
# endif()

# if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/gpt2_Opset18.onnx)
#     execute_process(
#         COMMAND ${WGET_EXECUTABLE} -q -O gpt2_Opset18.onnx "https://github.com/onnx/models/raw/main/Generative_AI/skip/gpt2_Opset18_transformers/gpt2_Opset18.onnx"
#         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#     )
#     execute_process(
#         COMMAND ${ONNXSIM} gpt2_Opset18.onnx gpt2_Opset18.onnx
#         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#     )
# endif()

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/bert_Opset18.onnx
    COMMAND ${WGET_EXECUTABLE} -q -O bert_Opset18.onnx "https://github.com/onnx/models/raw/main/Natural_Language_Processing/bert_Opset18_transformers/bert_Opset18.onnx"
    COMMAND ${ONNXSIM} bert_Opset18.onnx bert_Opset18.onnx
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gpt2_Opset18.onnx
    COMMAND ${WGET_EXECUTABLE} -q -O gpt2_Opset18.onnx "https://github.com/onnx/models/raw/main/Generative_AI/skip/gpt2_Opset18_transformers/gpt2_Opset18.onnx"
    COMMAND ${ONNXSIM} gpt2_Opset18.onnx gpt2_Opset18.onnx
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(
    download_bert
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/bert_Opset18.onnx
)

add_custom_target(
    download_gpt2
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/gpt2_Opset18.onnx
)

set(BERT_MODEL_PATH ${CMAKE_CURRENT_BINARY_DIR}/bert_Opset18.onnx)
set(GPT2_MODEL_PATH ${CMAKE_CURRENT_BINARY_DIR}/gpt2_Opset18.onnx)

add_subdirectory(model)
add_subdirectory(node)
if(${BUILD_PYTHON})
    add_subdirectory(python)
endif()
add_subdirectory(tensor)
add_subdirectory(utils)

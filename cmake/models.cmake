find_program(ONNXSIM NAMES onnxsim REQUIRED)

find_package(Wget REQUIRED)

if(NOT TARGET prepare)
    add_custom_target(
        prepare
    )
endif()

if (NOT TARGET prepare_bert)
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/models/bert_Opset18.onnx
        COMMAND ${WGET_EXECUTABLE} -q -O bert_Opset18.onnx "https://github.com/onnx/models/raw/main/Natural_Language_Processing/bert_Opset18_transformers/bert_Opset18.onnx"
        COMMAND ${ONNXSIM} bert_Opset18.onnx bert_Opset18.onnx
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/models
    )
    add_custom_target(
        prepare_bert
        DEPENDS ${CMAKE_BINARY_DIR}/models/bert_Opset18.onnx
    )
    add_dependencies(prepare prepare_bert)
endif()

if (NOT TARGET prepare_gpt2)
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/models/gpt2_Opset18.onnx
        COMMAND ${WGET_EXECUTABLE} -q -O gpt2_Opset18.onnx "https://github.com/onnx/models/raw/main/Generative_AI/skip/gpt2_Opset18_transformers/gpt2_Opset18.onnx"
        COMMAND ${ONNXSIM} gpt2_Opset18.onnx gpt2_Opset18.onnx
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/models
    )
    add_custom_target(
        prepare_gpt2
        DEPENDS ${CMAKE_BINARY_DIR}/models/gpt2_Opset18.onnx
    )
    add_dependencies(prepare prepare_gpt2)
endif()

set(BERT_MODEL_PATH ${CMAKE_BINARY_DIR}/models/bert_Opset18.onnx)
set(GPT2_MODEL_PATH ${CMAKE_BINARY_DIR}/models/gpt2_Opset18.onnx)
add_compile_definitions(BERT_MODEL_PATH="${BERT_MODEL_PATH}" GPT2_MODEL_PATH="${GPT2_MODEL_PATH}")

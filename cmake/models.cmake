find_program(ONNXSIM NAMES onnxsim REQUIRED)

find_package(Wget REQUIRED)

if(NOT TARGET prepare)
    add_custom_target(
        prepare
    )
endif()

if(NOT TARGET prepare_bert)
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

if(NOT TARGET prepare_convbert)
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/models/convbert_Opset18.onnx
        COMMAND ${WGET_EXECUTABLE} -q -O convbert_Opset18.onnx "https://github.com/onnx/models/raw/refs/heads/main/Natural_Language_Processing/convbert_Opset18_transformers/convbert_Opset18.onnx"
        COMMAND ${ONNXSIM} convbert_Opset18.onnx convbert_Opset18.onnx
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/models
    )
    add_custom_target(
        prepare_convbert
        DEPENDS ${CMAKE_BINARY_DIR}/models/convbert_Opset18.onnx
    )
    add_dependencies(prepare prepare_convbert)
endif()

if(NOT TARGET prepare_gptneox)
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/models/gptneox_Opset18.onnx
        COMMAND ${WGET_EXECUTABLE} -q -O gptneox_Opset18.onnx "https://github.com/onnx/models/raw/refs/heads/main/Generative_AI/gptneox_Opset18_transformers/gptneox_Opset18.onnx"
        COMMAND ${ONNXSIM} gptneox_Opset18.onnx gptneox_Opset18.onnx
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/models
    )
    add_custom_target(
        prepare_gptneox
        DEPENDS ${CMAKE_BINARY_DIR}/models/gptneox_Opset18.onnx
    )
    add_dependencies(prepare prepare_gptneox)
endif()

if(NOT TARGET prepare_ibert)
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/models/ibert_Opset17.onnx
        COMMAND ${WGET_EXECUTABLE} -q -O ibert_Opset17.onnx "https://github.com/onnx/models/raw/refs/heads/main/Natural_Language_Processing/ibert_Opset17_transformers/ibert_Opset17.onnx"
        COMMAND ${ONNXSIM} ibert_Opset17.onnx ibert_Opset17.onnx
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/models
    )
    add_custom_target(
        prepare_ibert
        DEPENDS ${CMAKE_BINARY_DIR}/models/ibert_Opset17.onnx
    )
    add_dependencies(prepare prepare_ibert)
endif()

if(NOT TARGET prepare_vgg)
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/models/vgg16-12.onnx
        COMMAND ${WGET_EXECUTABLE} -q -O vgg16-12.onnx "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/vgg/model/vgg16-12.onnx"
        COMMAND ${ONNXSIM} vgg16-12.onnx vgg16-12.onnx
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/models
    )
    add_custom_target(
        prepare_vgg
        DEPENDS ${CMAKE_BINARY_DIR}/models/vgg16-12.onnx
    )
    add_dependencies(prepare prepare_vgg)
endif()

set(BERT_MODEL_PATH ${CMAKE_BINARY_DIR}/models/bert_Opset18.onnx)
set(CONVBERT_MODEL_PATH ${CMAKE_BINARY_DIR}/models/convbert_Opset18.onnx)
set(GPTNEOX_MODEL_PATH ${CMAKE_BINARY_DIR}/models/gptneox_Opset18.onnx)
set(IBERT_MODEL_PATH ${CMAKE_BINARY_DIR}/models/ibert_Opset17.onnx)
set(VGG_MODEL_PATH ${CMAKE_BINARY_DIR}/models/vgg16-12.onnx)
add_compile_definitions(BERT_MODEL_PATH="${BERT_MODEL_PATH}" CONVBERT_MODEL_PATH="${CONVBERT_MODEL_PATH}" GPTNEOX_MODEL_PATH="${GPTNEOX_MODEL_PATH}" IBERT_MODEL_PATH="${IBERT_MODEL_PATH}" VGG_MODEL_PATH="${VGG_MODEL_PATH}")

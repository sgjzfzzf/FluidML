include(${CMAKE_SOURCE_DIR}/cmake/models.cmake)

set(
    TestIds
    "add0"
    "add1"
    "cast"
    "concat"
    "constant_of_shape"
    "conv0"
    "conv1"
    "conv2"
    "cum_sum"
    "div0"
    "div1"
    "dropout"
    "equal"
    "erf"
    "flatten"
    "gather_add_add"
    "gather0"
    "gather1"
    "gather2"
    "gemm"
    "layer_normalization"
    "matmul0"
    "matmul1"
    "maxpool"
    "mul0"
    "mul1"
    "neg"
    "not"
    "pad"
    "pow"
    "reduce_mean"
    "relu"
    "reshape"
    "slice"
    "softmax"
    "sqrt"
    "sub0"
    "sub1"
    "tanh"
    "transpose"
    "unsqueeze"
    "where"
)

set(PyEnvStr)
foreach(TestId ${TestIds})
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/models/${TestId}.onnx
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${TestId}.py ${CMAKE_BINARY_DIR}/models/${TestId}.onnx
        DEPENDS ${TestId}.py
    )
    add_custom_target(
        prepare_${TestId}
        DEPENDS ${CMAKE_BINARY_DIR}/models/${TestId}.onnx
    )
    add_dependencies(prepare prepare_${TestId})
    add_compile_definitions(ONNX_${TestId}_PATH="${CMAKE_BINARY_DIR}/models/${TestId}.onnx")
    list(APPEND PyEnvStr "ONNX_${TestId}_PATH=${CMAKE_BINARY_DIR}/models/${TestId}.onnx")
endforeach()

if(${BUILD_PYTHON} STREQUAL "ON")
    add_test(NAME PyNodeTest COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/node_test.py WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(PyNodeTest
        PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/src:$ENV{PYTHONPATH};${PyEnvStr}"
    )
endif()

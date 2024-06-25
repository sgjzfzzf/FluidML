set(
    TestIds
    "add0"
    "add1"
    "cast"
    "constant_of_shape"
    "div"
    "equal"
    "erf"
    "gather_add_add"
    "gather0"
    "gather1"
    "gemm"
    "layer_normalization"
    "matmul0"
    "matmul1"
    "mul0"
    "mul1"
    "pow"
    "reshape"
    "softmax"
    "split"
    "sub"
    "tanh"
    "transpose"
    "unsqueeze"
    "where"
)

set(OpModels)

set(PyEnvStr)

foreach(TestId ${TestIds})
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TestId}.onnx
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${TestId}.py ${CMAKE_CURRENT_BINARY_DIR}/${TestId}.onnx
        DEPENDS ${TestId}.py
    )
    list(APPEND OpModels ${CMAKE_CURRENT_BINARY_DIR}/${TestId}.onnx)
    add_compile_definitions(ONNX_${TestId}_PATH="${CMAKE_CURRENT_BINARY_DIR}/${TestId}.onnx")
    list(APPEND PyEnvStr "ONNX_${TestId}_PATH=${CMAKE_CURRENT_BINARY_DIR}/${TestId}.onnx")
endforeach()

add_executable(node_test node_test.cpp)
add_custom_target(NodeOnnxDepsTarget DEPENDS ${OpModels})
add_dependencies(node_test NodeOnnxDepsTarget)
target_link_libraries(node_test ${GTEST_BOTH_LIBRARIES} ${PROJECT_NAME}Static)
add_test(NAME NodeTest COMMAND node_test WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(PyNodeOnnxDepsTarget DEPENDS ${OpModels})
add_test(NAME PyNodeTest COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/node_test.py WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(PyNodeTest
    PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/src:$ENV{PYTHONPATH};${PyEnvStr}"
)

add_executable(graph_test graph_test.cpp)
target_link_libraries(graph_test ${GTEST_BOTH_LIBRARIES} ${PROJECT_NAME}Static)
add_test(NAME GraphTest COMMAND ./graph_test WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

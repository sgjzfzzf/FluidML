include(${CMAKE_SOURCE_DIR}/cmake/models.cmake)

add_test(
    NAME PyImportTest 
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import_test.py
)

add_test(
    NAME PyModelTest 
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/model_test.py
)

set_tests_properties(
    PyImportTest PyModelTest
    PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/src:$ENV{PYTHONPATH};BERT_MODEL_PATH=${BERT_MODEL_PATH};CONVBERT_MODEL_PATH=${CONVBERT_MODEL_PATH};EFFICIENTNET_MODEL_PATH=${EFFICIENTNET_MODEL_PATH};GPTNEOX_MODEL_PATH=${GPTNEOX_MODEL_PATH};IBERT_MODEL_PATH=${IBERT_MODEL_PATH};VGG_MODEL_PATH=${VGG_MODEL_PATH}"
)

add_subdirectory(node)

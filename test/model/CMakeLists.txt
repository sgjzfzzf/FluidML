add_compile_definitions(BERT_MODEL_PATH="${BERT_MODEL_PATH}")
add_compile_definitions(GPT2_MODEL_PATH="${GPT2_MODEL_PATH}")

add_executable(model_test model_test.cpp)
add_dependencies(model_test download_bert libfmt libglog)
target_link_libraries(model_test glog ${GTEST_BOTH_LIBRARIES} ${PROJECT_NAME}Static)
add_test(NAME ModelTest COMMAND ./model_test --log_dir=${CMAKE_CURRENT_BINARY_DIR} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

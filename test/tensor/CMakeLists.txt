add_executable(meta_test meta_test.cpp)
target_link_libraries(meta_test ${GTEST_BOTH_LIBRARIES} LIB_STATIC)
add_test(NAME MetaTest COMMAND ./meta_test)


add_executable(tensor_test tensor_test.cpp)
target_link_libraries(tensor_test ${GTEST_BOTH_LIBRARIES} LIB_STATIC)
add_test(NAME TensorTest COMMAND ./tensor_test)

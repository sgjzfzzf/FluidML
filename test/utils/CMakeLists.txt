add_executable(isa_test isa_test.cpp)
target_link_libraries(isa_test ${GTEST_BOTH_LIBRARIES})
add_test(NAME IsaTest COMMAND ./isa_test)

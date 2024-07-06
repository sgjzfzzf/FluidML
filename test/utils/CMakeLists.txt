add_executable(gen_test gen_test.cpp)
target_link_libraries(gen_test ${GTEST_BOTH_LIBRARIES} ${PROJECT_NAME}Static)
add_test(NAME GenShapeTest COMMAND ./gen_test)

add_executable(isa_test isa_test.cpp)
target_link_libraries(isa_test ${GTEST_BOTH_LIBRARIES})
add_test(NAME IsaTest COMMAND ./isa_test)

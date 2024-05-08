#include "structure/tensor/tensor.h"
#include "gtest/gtest.h"

using namespace cpu_transformers;

TEST(TensorTest, BasicTest) {
  Tensor tensor(Type::FLOAT32, {1, 2, 3}, {0, 1, 2, 3, 4, 5});
  tensor.Get({0, 1, 2}) = 6;
  EXPECT_EQ(tensor.Get({0, 1, 2}), 6);
  tensor.Get({0, 0, 0}) = 7;
  EXPECT_EQ(tensor.Get({0, 0, 0}), 7);
}

TEST(TensorTest, EqualTest) {
  Tensor tensor1(Type::FLOAT32, {1, 2, 3}, {0, 1, 2, 3, 4, 5});
  Tensor tensor2(Type::FLOAT32, {1, 2, 3}, {0, 1, 2, 3, 4, 5});
  EXPECT_EQ(tensor1, tensor2);
  tensor2.Get({0, 1, 2}) = 6;
  EXPECT_NE(tensor1, tensor2);
}

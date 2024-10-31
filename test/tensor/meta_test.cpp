#include "structure/tensor/meta.h"
#include "gtest/gtest.h"

using namespace fluidml;

TEST(MetaTest, BroadcastTest) {
  Meta meta0(Type::kFloat32, {1, 1, 3});
  Meta meta1(Type::kFloat32, {2, 3});
  std::optional<Meta> output_meta_opt =
      BroadcastShape(meta0, meta1, Type::kFloat32);
  ASSERT_TRUE(output_meta_opt.has_value());
  Meta output_meta = output_meta_opt.value();
  EXPECT_EQ(output_meta.GetType(), Type::kFloat32);
  EXPECT_EQ(output_meta.GetShape(), std::vector<int64_t>({1, 2, 3}));
}

TEST(MetaTest, BroadcastMatMulTest) {
  Meta meta0(Type::kFloat32, {2, 1, 3});
  Meta meta1(Type::kFloat32, {3, 2});
  std::optional<Meta> output_meta_opt =
      BroadcastMatMulShape(meta0, meta1, Type::kFloat32);
  ASSERT_TRUE(output_meta_opt.has_value());
  Meta output_meta = output_meta_opt.value();
  EXPECT_EQ(output_meta.GetType(), Type::kFloat32);
  EXPECT_EQ(output_meta.GetShape(), std::vector<int64_t>({2, 1, 2}));
}

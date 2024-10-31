#include "utils/utils.h"
#include "gtest/gtest.h"
#include <unordered_set>

TEST(GenTest, OrderTest) {
  constexpr size_t num = 8;
  std::vector<std::vector<size_t>> arrays = fluidml::utils::GenAllOrders(num);
  size_t expected_elem_num = 1;
  for (size_t i = 0; i < num; ++i) {
    expected_elem_num *= num - i;
  }
  ASSERT_EQ(arrays.size(), expected_elem_num);
  for (const auto &array : arrays) {
    ASSERT_EQ(array.size(), num);
    std::unordered_set<size_t> set;
    for (size_t i : array) {
      ASSERT_TRUE(i < num && i >= 0);
      set.insert(i);
    }
    for (size_t i = 0; i < num; ++i) {
      ASSERT_TRUE(set.find(i) != set.end());
    }
  }
}

TEST(GenTest, IndexTest) {
  std::vector<int64_t> shape = {2, 3, 4};
  const size_t shape_len = shape.size();
  std::vector<std::vector<size_t>> indices =
      fluidml::utils::GenAllIndicesInOrder(shape);
  ASSERT_EQ(indices.size(), 2 * 3 * 4);
  int64_t cur_index = -1;
  for (const std::vector<size_t> &indices : indices) {
    ASSERT_EQ(indices.size(), shape_len);
    ASSERT_LE(indices[0], shape[0]);
    ASSERT_LE(indices[1], shape[1]);
    ASSERT_LE(indices[2], shape[2]);
    int64_t real_index = indices[0] * 12 + indices[1] * 4 + indices[2];
    ASSERT_EQ(cur_index + 1, real_index);
    cur_index = real_index;
  }
}

TEST(GenTest, StrideTest) {
  std::vector<int64_t> shape = {2, 3, 4};
  std::vector<int64_t> strides = fluidml::utils::GenStrides(shape);
  ASSERT_EQ(strides.size(), shape.size());
  ASSERT_EQ(strides[0], 12);
  ASSERT_EQ(strides[1], 4);
  ASSERT_EQ(strides[2], 1);
}

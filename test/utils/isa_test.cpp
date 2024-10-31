#include "utils/isa.hpp"
#include "gtest/gtest.h"

namespace {
class Base {
public:
  Base() = default;
  Base(const Base &) = default;
  Base(Base &&) = default;
  virtual ~Base() = default;
};

class Derived0 : public Base {
public:
  Derived0() = default;
  Derived0(const Derived0 &) = default;
  Derived0(Derived0 &&) = default;
};

class Derived1 : public Base {
public:
  Derived1() = default;
  Derived1(const Derived1 &) = default;
  Derived1(Derived1 &&) = default;
};
} // namespace

TEST(IsaTest, RawPointerTest) {
  using namespace fluidml;
  Derived0 derived;
  Base *base = &derived;
  ASSERT_TRUE(isa<Derived0>(base));
  ASSERT_FALSE(isa<Derived1>(base));
}

TEST(IsaTest, UniquePointerTest) {
  using namespace fluidml;

  std::unique_ptr<Base> derived = std::make_unique<Derived0>();
  ASSERT_TRUE(isa<Derived0>(derived));
  ASSERT_FALSE(isa<Derived1>(derived));
}

TEST(IsaTest, SharedPointerTest) {
  using namespace fluidml;

  std::shared_ptr<Base> derived = std::make_shared<Derived0>();
  ASSERT_TRUE(isa<Derived0>(derived));
  ASSERT_FALSE(isa<Derived1>(derived));
}

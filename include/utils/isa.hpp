#ifndef FLUIDML_UTILS_ISA_HPP_
#define FLUIDML_UTILS_ISA_HPP_

#include <memory>
#include <type_traits>

namespace fluidml {

template <typename Derived, typename Base,
          typename = std::enable_if<std::is_base_of_v<Base, Derived>>>
bool isa(const Base *base) noexcept {
  return dynamic_cast<const Derived *>(base) != nullptr;
}

template <typename Derived, typename Base>
bool isa(const std::unique_ptr<Base> &unique_ptr) noexcept {
  return isa<Derived>(unique_ptr.get());
}

template <typename Derived, typename Base>
bool isa(const std::shared_ptr<Base> &shared_ptr) noexcept {
  return isa<Derived>(shared_ptr.get());
}

} // namespace fluidml

#endif

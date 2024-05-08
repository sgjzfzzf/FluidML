#include "structure/memory/info.h"
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace memory {
Info::Info(std::string &&name, size_t start, size_t end, size_t size)
    : name_(std::move(name)), start_(start), end_(end), size_(size) {
#ifdef DEBUG
  assert(start_ <= end_);
#endif
}

const std::string &Info::GetName() const { return name_; }

size_t Info::GetStart() const { return start_; }

size_t Info::GetEnd() const { return end_; }

size_t Info::GetSize() const { return size_; }

bool PlainInfos::IsEmpty() const { return infos_.empty(); }

void PlainInfos::Push(Info &&info) { infos_.push_back(std::move(info)); }

Info PlainInfos::Pop() {
#ifdef DEBUG
  assert(!infos_.empty());
#endif
  Info info = std::move(infos_.front());
  infos_.pop_front();
  return std::move(info);
}

bool GreedyInfos::InfoCompare::operator()(const Info &lhs,
                                          const Info &rhs) const {
  return lhs.GetSize() < rhs.GetSize();
}

bool GreedyInfos::IsEmpty() const { return infos_.empty(); }

void GreedyInfos::Push(Info &&info) { infos_.push(std::move(info)); }

Info GreedyInfos::Pop() {
#ifdef DEBUG
  assert(!infos_.empty());
#endif
  Info info = std::move(infos_.top());
  infos_.pop();
  return std::move(info);
}
} // namespace memory
} // namespace cpu_transformers

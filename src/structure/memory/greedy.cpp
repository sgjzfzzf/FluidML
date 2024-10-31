#include "structure/memory/greedy.h"
#include "structure/memory/index.h"
#include <memory>
#ifdef DEBUG
#include <cassert>
#endif

namespace {

class Manager {
public:
  struct Life {
    size_t start_;
    size_t end_;
    size_t life_start_;
    size_t life_end_;
  };
  Manager() = default;
  Manager(const Manager &) = delete;
  Manager(Manager &&) = default;
  ~Manager() = default;
  void Put(Life &&life);
  std::vector<Life> Filter(size_t life_interval) const;
  std::vector<Life> Filter(size_t life_start, size_t life_end) const;
#ifdef DEBUG
  bool Check() const;
#endif

private:
  std::vector<Life> lives_;
};

class Memory {
public:
  // The relationship of `start` and `end` is [start, end).
  struct Slice {
    size_t start_;
    size_t end_;
    std::shared_ptr<Slice> next_;
  };
  Memory();
  Memory(const Memory &) = delete;
  Memory(Memory &&) = default;
  ~Memory() = default;
  bool Busy(size_t start, size_t end);
  size_t Alloc(size_t size);
#ifdef DEBUG
  bool Check() const;
#endif

private:
  std::shared_ptr<Slice> head_;
};

void Manager::Put(Life &&life) { lives_.push_back(std::move(life)); }

std::vector<Manager::Life> Manager::Filter(size_t life_interval) const {
  return Filter(life_interval, life_interval);
}

std::vector<Manager::Life> Manager::Filter(size_t life_start,
                                           size_t life_end) const {
  std::vector<Life> filtered;
  for (const Life &life : lives_) {
    if (life.life_start_ <= life_end && life.life_end_ >= life_start) {
      filtered.push_back(life);
    }
  }
  return filtered;
}

#ifdef DEBUG
bool Manager::Check() const {
  size_t max = 0;
  for (const Life &life : lives_) {
    max = std::max(max, life.life_end_);
  }
  for (size_t i = 0; i < max; ++i) {
    std::vector<Life> filtered = Filter(i);
    const size_t len = filtered.size();
    for (size_t j = 0; j < len; ++j) {
      for (size_t k = j + 1; k < len; ++k) {
        const Life &lhs = filtered[j];
        const Life &rhs = filtered[k];
        if (lhs.start_ < rhs.end_ && lhs.end_ > rhs.start_) {
          return false;
        }
      }
    }
  }
  return true;
}
#endif

Memory::Memory() : head_(nullptr) {}

bool Memory::Busy(size_t start, size_t end) {
  if (start >= end) {
    return false;
  }
  if (head_) {
    if (end < head_->start_) {
      std::shared_ptr<Slice> slice = std::make_shared<Slice>();
      slice->start_ = start;
      slice->end_ = end;
      slice->next_ = head_;
      head_ = slice;
    } else if (start <= head_->end_ && end >= head_->start_) {
      head_->start_ = std::min(head_->start_, start);
      head_->end_ = std::max(head_->end_, end);
      std::shared_ptr<Slice> next = head_->next_;
      if (next && next->start_ <= end) {
        head_->end_ = next->end_;
        head_->next_ = next->next_;
      }
    } else if (end >= head_->end_) {
      std::shared_ptr<Slice> current = head_;
      while (current) {
        const size_t cur_start = current->start_;
        const size_t cur_end = current->end_;
        std::shared_ptr<Slice> next = current->next_;
        if (next == nullptr || next->end_ >= start) {
          while (next && next->start_ < end) {
            start = std::min(start, next->start_);
            end = std::max(end, next->end_);
            next = next->next_;
          }
#ifdef DEBUG
          assert(start <= end);
          assert(next == nullptr || current->end_ < next->start_);
#endif
          if ((next == nullptr || next->start_ > end) &&
              current->end_ >= start) {
            current->end_ = end;
            current->next_ = next;
          } else if (next && next->start_ <= end && current->end_ < start) {
            next->start_ = start;
            current->next_ = next;
          } else if ((next == nullptr || next->start_ > end) &&
                     current->end_ < start) {
            std::shared_ptr<Slice> slice = std::make_shared<Slice>();
            slice->start_ = start;
            slice->end_ = end;
            slice->next_ = next;
            current->next_ = slice;
          } else if (next && current->end_ >= start && next->start_ <= end) {
            current->end_ = next->end_;
            current->next_ = next->next_;
          } else {
#ifdef DEBUG
            assert(false && "unreachable branch");
#else
            __builtin_unreachable();
#endif
          }
          break;
        } else {
          current = next;
        }
      }
    } else {
#ifdef DEBUG
      assert(false && "unreachable branch");
#else
      __builtin_unreachable();
#endif
    }
  } else {
    head_ = std::make_unique<Slice>();
    head_->start_ = start;
    head_->end_ = end;
  }
  return true;
}

size_t Memory::Alloc(size_t size) {
  size_t start = -1;
  size_t end = -1;
  size_t most_matched_size = -1;
  if (head_) {
    std::shared_ptr<Slice> current = head_;
    while (current) {
#ifdef DEBUG
      assert(current->start_ < current->end_);
      assert(current->next_ == nullptr ||
             current->end_ <= current->next_->start_);
#endif
      std::shared_ptr<Slice> next = current->next_;
      if (next) {
        const size_t cur_end = current->end_;
        const size_t next_start = next->start_;
        const size_t free_size = next_start - cur_end;
        if (free_size >= size && free_size < most_matched_size) {
          start = cur_end;
          end = next_start;
          most_matched_size = free_size;
        }
      } else if (most_matched_size == -1) {
        start = current->end_;
        end = current->end_ + size;
      }
      current = current->next_;
    }
  } else {
    start = 0;
    end = size;
  }
#ifdef DEBUG
  assert(start != -1);
  assert(end != -1);
  assert(
#endif
      Busy(start, end)
#ifdef DEBUG
  )
#endif
      ;
  return start;
}

#ifdef DEBUG
bool Memory::Check() const {
  for (std::shared_ptr<Slice> current = head_; current;
       current = current->next_) {
    if (current->start_ >= current->end_ ||
        current->next_ && current->end_ >= current->next_->start_) {
      return false;
    }
  }
  return true;
}
#endif

} // namespace

namespace fluidml {
namespace memory {

Index GreedyPlan::Run(Infos &info) const {
  Index index;
  Manager manager;
  while (!info.IsEmpty()) {
    Memory memory;
    Info current = info.Pop();
    const std::string &name = current.GetName();
    const size_t size = current.GetSize();
    const size_t life_start = current.GetStart();
    const size_t life_end = current.GetEnd() + 1;
    std::vector<Manager::Life> lives = manager.Filter(life_start, life_end);
    for (const Manager::Life &life : lives) {
      memory.Busy(life.start_, life.end_);
#ifdef DEBUG
      assert(memory.Check());
#endif
    }
    const size_t start = memory.Alloc(size);
#ifdef DEBUG
    if (!memory.Check()) {
      assert(false);
    }
    assert(start >= 0);
#endif
    manager.Put({start, start + size, life_start, life_end});
    index.Set(name, start, size);
  }
#ifdef DEBUG
  assert(manager.Check());
#endif
  return index;
}

} // namespace memory
} // namespace fluidml

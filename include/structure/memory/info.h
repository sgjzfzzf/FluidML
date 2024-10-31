#ifndef FLUIDML_STRUCTURE_MEMORY_INFO_H_
#define FLUIDML_STRUCTURE_MEMORY_INFO_H_

#include <list>
#include <queue>
#include <string>

namespace fluidml {
namespace memory {

class Info {
public:
  Info(std::string &&name, size_t start, size_t end, size_t size);
  Info(const Info &) = default;
  Info(Info &&) = default;
  Info &operator=(const Info &) = delete;
  Info &operator=(Info &&) = default;
  ~Info() = default;
  const std::string &GetName() const;
  size_t GetStart() const;
  size_t GetEnd() const;
  size_t GetSize() const;

private:
  std::string name_;
  size_t start_;
  size_t end_;
  size_t size_;
};

class Infos {
public:
  Infos() = default;
  Infos(const Infos &) = delete;
  Infos(Infos &&) = default;
  virtual ~Infos() = default;
  virtual bool IsEmpty() const = 0;
  virtual void Push(Info &&info) = 0;
  virtual Info Pop() = 0;
};

class PlainInfos : public Infos {
public:
  PlainInfos() = default;
  PlainInfos(const PlainInfos &) = delete;
  PlainInfos(PlainInfos &&) = default;
  ~PlainInfos() = default;
  bool IsEmpty() const override;
  void Push(Info &&info) override;
  Info Pop() override;

private:
  std::list<Info> infos_;
};

class GreedyInfos : public Infos {
public:
  struct InfoCompare {
    bool operator()(const Info &lhs, const Info &rhs) const;
  };
  GreedyInfos() = default;
  GreedyInfos(const GreedyInfos &) = delete;
  GreedyInfos(GreedyInfos &&) = default;
  ~GreedyInfos() = default;
  bool IsEmpty() const override;
  void Push(Info &&info) override;
  Info Pop() override;

private:
  std::priority_queue<Info, std::vector<Info>, InfoCompare> infos_;
};

} // namespace memory
} // namespace fluidml

#endif

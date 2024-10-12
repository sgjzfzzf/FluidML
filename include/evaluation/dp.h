#ifndef CPU_TRANSFORMERS_EVALUATION_DP_H_
#define CPU_TRANSFORMERS_EVALUATION_DP_H_

#include "nlohmann/json_fwd.hpp"
#include "structure/flow/flow.h"
#include <ostream>
#include <string>

namespace cpu_transformers {
namespace evaluation {

class DynamicProgrammingPlan {
public:
  DynamicProgrammingPlan() = default;
  DynamicProgrammingPlan(
      std::unordered_map<std::string, std::vector<size_t>> &&plan);
  DynamicProgrammingPlan(const DynamicProgrammingPlan &plan) = delete;
  DynamicProgrammingPlan(DynamicProgrammingPlan &&plan) = default;
  DynamicProgrammingPlan &
  operator=(const DynamicProgrammingPlan &plan) = delete;
  DynamicProgrammingPlan &operator=(DynamicProgrammingPlan &&plan) = default;
  virtual ~DynamicProgrammingPlan() = default;
  bool HasLayout(const std::string &name) const;
  const std::vector<size_t> &GetLayout(const std::string &name) const;
  friend DynamicProgrammingPlan Merge(const DynamicProgrammingPlan &lhs,
                                      const DynamicProgrammingPlan &rhs);
  nlohmann::json ToJson() const;
  friend std::ostream &operator<<(std::ostream &os,
                                  const DynamicProgrammingPlan &plan);

private:
  std::unordered_map<std::string, std::vector<size_t>> plan_;
};

class DynamicProgrammingTable {
public:
  virtual DynamicProgrammingPlan Run() = 0;
  static std::shared_ptr<DynamicProgrammingTable> Make(const flow::Flow &flow);

protected:
  DynamicProgrammingTable() = default;
  DynamicProgrammingTable(const DynamicProgrammingTable &table) = delete;
  DynamicProgrammingTable(DynamicProgrammingTable &&table) = default;
  virtual ~DynamicProgrammingTable() = default;
};

} // namespace evaluation
} // namespace cpu_transformers

#endif

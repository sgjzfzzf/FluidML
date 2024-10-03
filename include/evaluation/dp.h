#ifndef CPU_TRANSFORMERS_EVALUATION_DP_H_
#define CPU_TRANSFORMERS_EVALUATION_DP_H_

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
  const std::vector<size_t> &GetLayout(const std::string &name) const;
  friend DynamicProgrammingPlan Merge(const DynamicProgrammingPlan &lhs,
                                      const DynamicProgrammingPlan &rhs);
#ifdef DEBUG
  friend std::ostream &operator<<(std::ostream &os,
                                  const DynamicProgrammingPlan &plan);
#endif

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
};

} // namespace evaluation
} // namespace cpu_transformers

#endif
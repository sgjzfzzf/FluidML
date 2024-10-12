#ifndef CPU_TRANSFORMERS_EVALUATION_FWD_H_
#define CPU_TRANSFORMERS_EVALUATION_FWD_H_

namespace cpu_transformers {
namespace evaluation {

class DynamicProgrammingPlan;
class DynamicProgrammingTable;

class KernelEval;
class SingleInputKernelEval;
class SingleInputWithoutBufferKernelEval;
class SingleInputWithBufferKernelEval;
class DoubleInputsKernelEval;
class DoubleInputsWithoutBufferKernelEval;

} // namespace evaluation
} // namespace cpu_transformers

#endif

#ifndef FLUIDML_EVALUATION_FWD_H_
#define FLUIDML_EVALUATION_FWD_H_

namespace fluidml {
namespace evaluation {

class DynamicProgrammingPlan;
class DynamicProgrammingTable;

class KernelEval;
class SingleInputKernelEval;
class SingleInputWithoutBufferKernelEval;
class SingleInputWithBufferKernelEval;
class DoubleInputsKernelEval;
class DoubleInputsWithoutBufferKernelEval;

class Factory;

} // namespace evaluation
} // namespace fluidml

#endif

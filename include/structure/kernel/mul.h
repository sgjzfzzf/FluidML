#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_MUL_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_MUL_H_

#include "structure/kernel/kernel.h"
#include "utils/float.h"
#include "utils/type.h"

namespace cpu_transformers {
namespace kernel {

class MulKernel : virtual public Kernel {
public:
  MulKernel() = default;
  MulKernel(const MulKernel &other) = delete;
  MulKernel(MulKernel &&other) = default;
  virtual ~MulKernel() = default;
};

class MulConstantKernel : public SingleInputWithoutBufferKernel,
                          public MulKernel {
public:
  MulConstantKernel(Type type, float64_t constant);
  MulConstantKernel(const MulConstantKernel &other) = delete;
  MulConstantKernel(MulConstantKernel &&other) = default;
  virtual ~MulConstantKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] = "MulConstantKernel";
  const Type type_;
  const float64_t constant_;
};

class MulCommonKernel : public DoubleInputsWithoutBufferKernel,
                        public MulKernel {
public:
  MulCommonKernel() = default;
  MulCommonKernel(const MulCommonKernel &other) = delete;
  MulCommonKernel(MulCommonKernel &&other) = default;
  virtual ~MulCommonKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] = "MulCommonKernel";
};

class MulConstantKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~MulConstantKernelGenerator() = default;
  virtual std::shared_ptr<MulConstantKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<MulConstantKernelGenerator> Make(Type type,
                                                          float64_t constant);

protected:
  MulConstantKernelGenerator() = default;
  MulConstantKernelGenerator(const MulConstantKernelGenerator &generator) =
      delete;
  MulConstantKernelGenerator(MulConstantKernelGenerator &&generator) = default;
};

class MulCommonKernelGenerator
    : public DoubleInputsWithoutBufferKernelGenerator {
public:
  virtual ~MulCommonKernelGenerator() = default;
  virtual std::shared_ptr<MulCommonKernel>
  Yield(llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<MulCommonKernelGenerator> Make();

protected:
  MulCommonKernelGenerator() = default;
  MulCommonKernelGenerator(const MulCommonKernelGenerator &generator) = delete;
  MulCommonKernelGenerator(MulCommonKernelGenerator &&generator) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif

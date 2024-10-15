#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_GENERATOR_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_GENERATOR_H_

#include "structure/kernel/kernel/kernel.h"
#include <memory>

namespace cpu_transformers {
namespace kernel {

class KernelGenerator {
public:
  virtual ~KernelGenerator() = default;

protected:
  KernelGenerator() = default;
  KernelGenerator(const KernelGenerator &generator) = delete;
  KernelGenerator(KernelGenerator &&generator) = default;
};

class SingleInputKernelGenerator : public KernelGenerator {
public:
  virtual ~SingleInputKernelGenerator() = default;
  virtual std::shared_ptr<SingleInputKernel>
  YieldSingleInputKernel(llvm::ArrayRef<size_t> input_layout,
                         llvm::ArrayRef<size_t> output_layout) = 0;

protected:
  SingleInputKernelGenerator() = default;
  SingleInputKernelGenerator(const SingleInputKernelGenerator &generator) =
      delete;
  SingleInputKernelGenerator(SingleInputKernelGenerator &&generator) = default;
};

class DoubleInputsKernelGenerator : public KernelGenerator {
public:
  virtual ~DoubleInputsKernelGenerator() = default;
  virtual std::shared_ptr<DoubleInputsKernel>
  YieldDoubleInputsKernel(llvm::ArrayRef<size_t> lhs_layout,
                          llvm::ArrayRef<size_t> rhs_layout,
                          llvm::ArrayRef<size_t> output_layout) = 0;

protected:
  DoubleInputsKernelGenerator() = default;
  DoubleInputsKernelGenerator(const DoubleInputsKernelGenerator &generator) =
      delete;
  DoubleInputsKernelGenerator(DoubleInputsKernelGenerator &&generator) =
      default;
};

class SingleInputWithoutBufferKernelGenerator
    : public SingleInputKernelGenerator {
public:
  virtual ~SingleInputWithoutBufferKernelGenerator() = default;
  std::shared_ptr<SingleInputKernel>
  YieldSingleInputKernel(llvm::ArrayRef<size_t> input_layout,
                         llvm::ArrayRef<size_t> output) override;
  virtual std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(llvm::ArrayRef<size_t> input_layout,
                                      llvm::ArrayRef<size_t> output_layout) = 0;

protected:
  SingleInputWithoutBufferKernelGenerator() = default;
  SingleInputWithoutBufferKernelGenerator(
      const SingleInputWithoutBufferKernelGenerator &generator) = delete;
  SingleInputWithoutBufferKernelGenerator(
      SingleInputWithoutBufferKernelGenerator &&generator) = default;
};

class SingleInputWithBufferKernelGenerator : public SingleInputKernelGenerator {
public:
  virtual ~SingleInputWithBufferKernelGenerator() = default;
  std::shared_ptr<SingleInputKernel>
  YieldSingleInputKernel(llvm::ArrayRef<size_t> input_layout,
                         llvm::ArrayRef<size_t> output) override;
  virtual std::shared_ptr<SingleInputWithBufferKernel>
  YieldSingleInputWithBufferKernel(llvm::ArrayRef<size_t> input_layout,
                                   llvm::ArrayRef<size_t> output_layout) = 0;

protected:
  SingleInputWithBufferKernelGenerator() = default;
  SingleInputWithBufferKernelGenerator(
      const SingleInputWithBufferKernelGenerator &generator) = delete;
  SingleInputWithBufferKernelGenerator(
      SingleInputWithBufferKernelGenerator &&generator) = default;
};

class DoubleInputsWithoutBufferKernelGenerator
    : public DoubleInputsKernelGenerator {
public:
  virtual ~DoubleInputsWithoutBufferKernelGenerator() = default;
  std::shared_ptr<DoubleInputsKernel>
  YieldDoubleInputsKernel(llvm::ArrayRef<size_t> lhs_layout,
                          llvm::ArrayRef<size_t> rhs_layout,
                          llvm::ArrayRef<size_t> output_layout) override;
  virtual std::shared_ptr<DoubleInputsWithoutBufferKernel>
  YieldDoubleInputsWithoutBufferKernel(
      llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
      llvm::ArrayRef<size_t> output_layout) = 0;

protected:
  DoubleInputsWithoutBufferKernelGenerator() = default;
  DoubleInputsWithoutBufferKernelGenerator(
      const DoubleInputsWithoutBufferKernelGenerator &generator) = delete;
  DoubleInputsWithoutBufferKernelGenerator(
      DoubleInputsWithoutBufferKernelGenerator &&generator) = default;
};

class DoubleInputsWithBufferKernelGenerator
    : public DoubleInputsKernelGenerator {
public:
  virtual ~DoubleInputsWithBufferKernelGenerator() = default;
  std::shared_ptr<DoubleInputsKernel>
  YieldDoubleInputsKernel(llvm::ArrayRef<size_t> lhs_layout,
                          llvm::ArrayRef<size_t> rhs_layout,
                          llvm::ArrayRef<size_t> output_layout) override;
  virtual std::shared_ptr<DoubleInputsWithBufferKernel>
  YieldDoubleInputsWithBufferKernel(llvm::ArrayRef<size_t> lhs_layout,
                                    llvm::ArrayRef<size_t> rhs_layout,
                                    llvm::ArrayRef<size_t> output_layout) = 0;

protected:
  DoubleInputsWithBufferKernelGenerator() = default;
  DoubleInputsWithBufferKernelGenerator(
      const DoubleInputsWithBufferKernelGenerator &generator) = delete;
  DoubleInputsWithBufferKernelGenerator(
      DoubleInputsWithBufferKernelGenerator &&generator) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif

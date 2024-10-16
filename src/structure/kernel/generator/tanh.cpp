#include "structure/kernel/generator/tanh.h"

namespace cpu_transformers {
namespace kernel {

class TanhKernelGeneratorImpl : public TanhKernelGenerator {
public:
  TanhKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta);
  TanhKernelGeneratorImpl(const TanhKernelGeneratorImpl &generator) = delete;
  TanhKernelGeneratorImpl(TanhKernelGeneratorImpl &&generator) = default;
  virtual ~TanhKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<TanhKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;

private:
  const Meta input_meta_;
  const Meta output_meta_;
};

std::unique_ptr<TanhKernelGenerator>
TanhKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta) {
  return std::make_unique<TanhKernelGeneratorImpl>(std::move(input_meta),
                                                   std::move(output_meta));
}

TanhKernelGeneratorImpl::TanhKernelGeneratorImpl(Meta &&input_meta,
                                                 Meta &&output_meta)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)) {
}

std::shared_ptr<SingleInputWithoutBufferKernel>
TanhKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<TanhKernel>
TanhKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                               llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<TanhKernel>();
}

const Meta &TanhKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &TanhKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string TanhKernelGeneratorImpl::GetKernelName() const {
  return TanhKernel::kKernelName;
}

} // namespace kernel
} // namespace cpu_transformers

#include "structure/kernel/generator/reshape.h"

namespace cpu_transformers {
namespace kernel {

class ReshapeKernelGeneratorImpl : public ReshapeKernelGenerator {
public:
  ReshapeKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta);
  ReshapeKernelGeneratorImpl(const ReshapeKernelGeneratorImpl &generator) =
      delete;
  ReshapeKernelGeneratorImpl(ReshapeKernelGeneratorImpl &&generator) = default;
  virtual ~ReshapeKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<ReshapeKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;

private:
  const Meta input_meta_;
  const Meta output_meta_;
};

std::unique_ptr<ReshapeKernelGenerator>
ReshapeKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta) {
  return std::make_unique<ReshapeKernelGeneratorImpl>(std::move(input_meta),
                                                      std::move(output_meta));
}

ReshapeKernelGeneratorImpl::ReshapeKernelGeneratorImpl(Meta &&input_meta,
                                                       Meta &&output_meta)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)) {
}

std::shared_ptr<SingleInputWithoutBufferKernel>
ReshapeKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<ReshapeKernel>
ReshapeKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                  llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<ReshapeKernel>();
}

const Meta &ReshapeKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &ReshapeKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string ReshapeKernelGeneratorImpl::GetKernelName() const {
  return ReshapeKernel::kKernelName;
}

} // namespace kernel
} // namespace cpu_transformers

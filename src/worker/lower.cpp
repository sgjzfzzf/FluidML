#include "worker/lower.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRVPass.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "structure/context/context.h"
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace worker {

class LowerImpl : public Lower {
public:
  LowerImpl(std::shared_ptr<context::Context> context);
  LowerImpl(const LowerImpl &lower) = delete;
  LowerImpl(LowerImpl &&lower) = delete;
  virtual ~LowerImpl() = default;
  void Run() override;

private:
  std::shared_ptr<context::Context> context_;
  mlir::PassManager pm_;
};

std::unique_ptr<Lower>
Lower::Make(std::shared_ptr<context::Context> &&context) {
  return std::make_unique<LowerImpl>(std::move(context));
}

LowerImpl::LowerImpl(std::shared_ptr<context::Context> context)
    : context_(context ? std::move(context) : context::Context::Make()),
      pm_(&context_->GetMLIRContext()) {
  pm_.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  pm_.addPass(mlir::createLowerAffinePass());
  pm_.addPass(mlir::createConvertSCFToCFPass());
  pm_.addPass(mlir::arith::createConstantBufferizePass());
  pm_.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm_.addPass(mlir::createConvertToLLVMPass());
  pm_.addPass(mlir::createConvertMathToLibmPass());
  pm_.addPass(mlir::createConvertToLLVMPass());
}

void LowerImpl::Run() {
  mlir::ModuleOp module = context_->GetModule();
#ifdef DEBUG
  assert(
#endif
      pm_.run(module).succeeded()
#ifdef DEBUG
  )
#endif
      ;
}

} // namespace worker
} // namespace cpu_transformers

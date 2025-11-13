/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertONNXToTOSA.cpp - ONNX dialects to TOSA lowering -------===//
//
// Copyright (c) 2022 Arm Limited.
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file implements the lowering of frontend operations to the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace onnx_mlir {

void populateONNXToTOSAConversionPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  // Math
  populateLoweringONNXElementwiseOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXReduceMeanOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  // populateLoweringONNXGemmOpToTOSAPattern(target, patterns, typeConverter, ctx);
  populateLoweringONNXSoftmaxOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXConvOpToTOSAPattern(target, patterns, typeConverter, ctx);
  // NN
  populateLoweringONNXMaxPoolSingleOutOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXAveragePoolOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  // Tensor
  populateLoweringONNXConstOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXReshapeOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXResizeOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
      
  //Custom

  populateLoweringONNXFlattenOpToTOSAPattern(
    target, patterns, typeConverter, ctx);

  populateLoweringONNXQLinearConvOpToTOSAPattern(
    target, patterns, typeConverter, ctx);

  populateLoweringONNXCustomOpToTOSAPattern(
    target, patterns, typeConverter, ctx);

  populateLoweringONNXGemmOpToTOSAPattern(
    target, patterns, typeConverter, ctx);

  populateLoweringONNXReluOpToTOSAPattern(
    target, patterns, typeConverter, ctx);

  populateLoweringONNXSigmoidOpToTOSAPattern(
      target, patterns, typeConverter, ctx);

  populateLoweringONNXSoftmaxOpToTOSAPattern(
    target, patterns, typeConverter, ctx);
    
  //populateLoweringONNXQuantizeLinearOpToTOSAPattern(target, patterns, typeConverter, ctx);



}

// Performs lowering to TOSA dialect
struct FrontendToTosaLoweringPass
    : public PassWrapper<FrontendToTosaLoweringPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "convert-onnx-to-tosa"; }

  Option<bool> useUnsigned{
    *this, "use-unsigned",
    llvm::cl::desc("Treat 8-bit tensors as unsigned"),
    llvm::cl::init(false)};

  StringRef getDescription() const override {
    return "Lower frontend ops to TOSA dialect.";
  }

  FrontendToTosaLoweringPass() = default;
  FrontendToTosaLoweringPass(const FrontendToTosaLoweringPass &pass)
      : PassWrapper<FrontendToTosaLoweringPass, OperationPass<ModuleOp>>() {}

  void runOnOperation() final;
};

void FrontendToTosaLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();
  // Define final conversion target
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  ConversionTarget target(*context);

  // We use the type converter to legalize types before any conversion patterns
  // are executed. This ensures that we do not need to trigger separate
  // conversion failures. Quantized types are not supported right now.
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) -> std::optional<Type> {
    if (isTOSASignedInt(type) || isTOSAFloat(type) || mlir::isa<NoneType>(type))
      return type;
    return std::nullopt;
  });
  typeConverter.addConversion([&](TensorType type) -> std::optional<Type> {
    if (typeConverter.isLegal(type.getElementType()))
      return type;
    return std::nullopt;
  });

  // Define legal dialects and operations
  target.addLegalDialect<mlir::tosa::TosaDialect, func::FuncDialect,
      mlir::arith::ArithDialect>();

  // Define patterns
  populateONNXToTOSAConversionPattern(target, patterns, typeConverter, context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }

  bool remove_QDQ = true;
  bool remove_transpose = true;

  if(remove_QDQ){
    OpBuilder builder(module.getContext());
    for (auto func : llvm::make_early_inc_range(module.getOps<func::FuncOp>())){
      if (func.getName() != "main_graph") continue;

      if (func.getNumArguments() != 1 || func.getFunctionType().getNumResults() != 1) continue;

      auto oldtype = func.getFunctionType();
      auto in = llvm::to_vector<4>(oldtype.getInputs());
      auto out = llvm::to_vector<4>(oldtype.getResults());
      auto inType = cast<RankedTensorType>(in[0]);
      auto outType = cast<RankedTensorType>(out[0]);
      auto inShape = inType.getShape();
      auto outShape = outType.getShape();

      auto newinput = RankedTensorType::get(inShape, IntegerType::get(context,8));
      auto newoutput = RankedTensorType::get(outShape,IntegerType::get(context,8));
      

      in[0] = newinput;
      out[0] = newoutput;
      auto newFuncType = mlir::FunctionType::get(func.getContext(), in, out);
      func.setType(newFuncType);

      Block &entry = func.getBody().front();
      BlockArgument arg = entry.getArgument(0);
      arg.setType(newinput);

    
      SmallVector<Operation*> toErase;
      for (Operation &op : llvm::make_early_inc_range(entry)){
        if (ONNXQuantizeLinearOp qop = dyn_cast<ONNXQuantizeLinearOp>(op))  {
          auto opers = qop.getOperands();
          for (Value ii : opers){
            if(isa<BlockArgument>(ii)){
              qop.getResult().replaceAllUsesWith(ii);
              qop.erase();
            }
          }
        }
        if (ONNXDequantizeLinearOp dqop = dyn_cast<ONNXDequantizeLinearOp>(op)){
          auto opers = dqop.getOperands();
          for (Value ii : opers){
            if(!isa<mlir::tosa::ConstOp>(ii.getDefiningOp())){
              dqop.getResult().replaceAllUsesWith(ii);
              dqop.erase();
            }
          }    
        }
        if(op.getResults().use_empty() && !isa<func::ReturnOp>(op))
          op.erase();        
      }
    }
  }
  
  if (remove_transpose){
    SmallVector<mlir::tosa::TransposeOp> transpose_remove;
    bool found_first = false;
    mlir::tosa::TransposeOp first_top;
    for (auto func : llvm::make_early_inc_range(module.getOps<func::FuncOp>())){
      Block &entry = func.getBody().front();
      for (Operation &op : llvm::make_early_inc_range(entry)){
        if (mlir::tosa::TransposeOp top = dyn_cast<mlir::tosa::TransposeOp>(op)){
          Value input = top.getInput1();
          if (isa<BlockArgument>(input)) continue;
          if (isa<mlir::tosa::ConstOp>(input.getDefiningOp())) continue;
          if (!found_first) {
            found_first = true;
            first_top = top;
          }
          else {
            found_first = false;
            transpose_remove.push_back(top);
            transpose_remove.push_back(first_top);
          }
        }
      }
    }
    for (mlir::tosa::TransposeOp top : transpose_remove){
      Value input = top.getInput1();
      top.replaceAllUsesWith(input);
      top.erase();
    }
  }
  
  module.walk([&](ONNXEntryPointOp ep) {
    ep.erase();
  });
}


std::unique_ptr<Pass> createConvertONNXToTOSAPass() {
  return std::make_unique<FrontendToTosaLoweringPass>();
}

} // namespace onnx_mlir

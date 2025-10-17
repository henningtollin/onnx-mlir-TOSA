#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"

using namespace mlir;

namespace onnx_mlir{

namespace {

class ONNXReluOpLoweringToTOSA : public OpConversionPattern<ONNXReluOp> { public:
    using OpConversionPattern<ONNXReluOp>::OpConversionPattern;
    using OpAdaptor = typename ONNXReluOp::Adaptor;
    LogicalResult matchAndRewrite(ONNXReluOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get the input and its type and shape.
    Value input = adaptor.getX();
    auto outType = cast<RankedTensorType>(op.getType());
    auto outShape = outType.getShape();
    auto outOnnxShape = outShape;
    ONNXDequantizeLinearOp dqX;
    ONNXQuantizeLinearOp qO;
    bool quantized = false;

    // Set the minimium and maximum limit of the clamp
    auto minval = rewriter.getI8IntegerAttr(0);
    auto maxval = rewriter.getI8IntegerAttr(std::numeric_limits<int8_t>::max());

    SmallVector<int64_t> outNCHW(outShape.begin(),outShape.end());
    SmallVector<int64_t> outNHWC;
    DenseI32ArrayAttr tosaPerms;


    // If the input is 4D inesert a transpose such that the input gets to tosa shape [N,H,W,C]
    // This will make it easier to remove the transposes later.
    if(outNCHW.size() == 4){        
        tosaPerms = DenseI32ArrayAttr::get(op.getContext(),{0,2,3,1});
        outNHWC = {outNCHW[0], outNCHW[2], outNCHW[3], outNCHW[1]};
        outType = RankedTensorType::get(outNHWC,outType.getElementType());
        outShape = outType.getShape();
    } else if (outNCHW.size() != 2){
        llvm::errs() << "This shape is not supported\n";
        rewriter.notifyMatchFailure(loc,"This shape is not supported");
    } 

    // If the we are in a QDQ network get the input before it is dequantized and use the a int8.
    if ((dqX = dyn_cast<ONNXDequantizeLinearOp>(input.getDefiningOp())) 
        && (qO = dyn_cast<ONNXQuantizeLinearOp>(op.getResult().getUsers().begin().getCurrent().getOperand()->getOwner()))){
        input = dqX.getX();
        outType = RankedTensorType::get(outShape,rewriter.getI8Type());        
        quantized = true;
    }

    // The transpose is inserted here
    if (outNCHW.size() == 4){
        input = rewriter.create<mlir::tosa::TransposeOp>(loc,outType,input,tosaPerms).getResult();
    }
    
    // Replace the Relu operator with a clamp
    Value clamp = rewriter.create<mlir::tosa::ClampOp>(loc, outType, input,minval,maxval).getResult();
    auto outElemType = cast<RankedTensorType>(clamp.getType()).getElementType();
    outType = RankedTensorType::get(outOnnxShape, outElemType);

    // If we inserted a transpose before use one more to get the input back to onnx format.
    DenseI32ArrayAttr onnxPerms;
    if (outNHWC.size() == 4) {
        onnxPerms = DenseI32ArrayAttr::get(op.getContext(),{0,3,1,2});
        clamp = rewriter.create<mlir::tosa::TransposeOp>(loc,outType, clamp,onnxPerms);
    } 
    
    // Replace the Relu with the clamp.
    if(quantized){
        rewriter.replaceOp(qO, clamp);
        rewriter.eraseOp(op);
        rewriter.eraseOp(dqX);
        return success();
    }
    
    rewriter.replaceOp(qO, clamp);
    
    return success();
    }
};
} // namespace

void populateLoweringONNXReluOpToTOSAPattern(
    ConversionTarget &target, RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
    patterns.insert<ONNXReluOpLoweringToTOSA>(typeConverter, ctx);
}

} // namspace onnx_mlir
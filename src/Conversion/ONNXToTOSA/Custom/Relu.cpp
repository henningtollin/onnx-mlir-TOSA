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
    Value input = adaptor.getX();
    auto outType = cast<RankedTensorType>(op.getType());
    auto outShape = outType.getShape();
    ONNXDequantizeLinearOp dqX;
    ONNXQuantizeLinearOp qO;
    bool quantized = false;
    auto minval = rewriter.getI8IntegerAttr(0);
    auto maxval = rewriter.getI8IntegerAttr(std::numeric_limits<int8_t>::max());

    if ((dqX = dyn_cast<ONNXDequantizeLinearOp>(input.getDefiningOp())) 
        && (qO = dyn_cast<ONNXQuantizeLinearOp>(op.getResult().getUsers().begin().getCurrent().getOperand()->getOwner()))){
        input = dqX.getX();
        outType = RankedTensorType::get(outShape,rewriter.getI8Type());
        outShape = outType.getShape();
        quantized = true;
    }
    Value clamp = rewriter.create<mlir::tosa::ClampOp>(loc, outType, input,minval,maxval).getResult();
    clamp.dump();
    if(quantized){
        rewriter.replaceOp(qO, clamp);
        rewriter.eraseOp(op);
        rewriter.eraseOp(dqX);
        return success();
    }

    
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
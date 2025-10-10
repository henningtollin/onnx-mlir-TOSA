


#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"



using namespace mlir;

namespace onnx_mlir {
namespace {

class ONNXFlattenOpLoweringToTosa : public OpConversionPattern<ONNXFlattenOp> { public:
    using OpConversionPattern<ONNXFlattenOp>::OpConversionPattern;
    using OpAdaptor = typename ONNXFlattenOp::Adaptor;
    LogicalResult matchAndRewrite(ONNXFlattenOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {

        Location loc = op.getLoc();

        //Get the data from the ONNX operator
        auto input = adaptor.getInput();
        auto outType = cast<RankedTensorType>(op.getType());
        auto outShape = outType.getShape();

        ONNXDequantizeLinearOp dqX;
        ONNXQuantizeLinearOp qO;
        bool quantized = false;

        if ((dqX = dyn_cast<ONNXDequantizeLinearOp>(input.getDefiningOp())) 
          && (qO = dyn_cast<ONNXQuantizeLinearOp>(op.getResult().getUsers().begin().getCurrent().getOperand()->getOwner()))){
            input = dqX.getX();
            outType = RankedTensorType::get(outShape,rewriter.getI8Type());
            quantized = true;
        }

        
        // Turn the shape of the output into a tosa constant shape
        Value ss = mlir::tosa::getTosaConstShape(rewriter,loc,outShape);

        // Create the flattened tosa tensor
        Value flatt = rewriter.create<mlir::tosa::ReshapeOp>(loc,outType, input, ss).getResult();

        if (quantized){
            rewriter.replaceOp(qO, flatt);
            rewriter.eraseOp(op);
            rewriter.eraseOp(dqX);
            return success();
        }
        // Lower to from onnx operator to tosa operator
        rewriter.replaceOp(op,flatt);

        return success();
    }
};


} //namspace
void populateLoweringONNXFlattenOpToTOSAPattern(ConversionTarget &target, RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *ctx) {
    patterns.insert<ONNXFlattenOpLoweringToTosa>(typeConverter, ctx);
}
} //namespace onnx_mlir
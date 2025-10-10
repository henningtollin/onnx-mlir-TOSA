


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

SmallVector<int64_t> AttributeVec(ArrayAttr attr){

    SmallVector<int64_t> att; 
    for (size_t i = 0; i < attr.size(); i++){
        int64_t a = cast<IntegerAttr>(attr[i]).getInt();
        att.push_back(a);
    }
    return att; 
    }

class ONNXMaxPoolSingleOutOpLoweringToTosa : public OpConversionPattern<ONNXMaxPoolSingleOutOp> { public:
    using OpConversionPattern<ONNXMaxPoolSingleOutOp>::OpConversionPattern;
    using OpAdaptor = typename ONNXMaxPoolSingleOutOp::Adaptor;
    LogicalResult matchAndRewrite(ONNXMaxPoolSingleOutOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {

        Location loc = op.getLoc();

        Value input = adaptor.getX();
        auto outType = cast<RankedTensorType>(op.getType());
        auto outShape = outType.getShape();
        auto kernel = op.getKernelShapeAttr();
        auto stride = op.getStridesAttr();
        auto pad = op.getPadsAttr();
        auto dilation = op.getDilationsAttr();
        
        ONNXDequantizeLinearOp dqX;
        ONNXQuantizeLinearOp qO;
        bool quantized = false;

        if ((dqX = dyn_cast<ONNXDequantizeLinearOp>(input.getDefiningOp())) 
          && (qO = dyn_cast<ONNXQuantizeLinearOp>(op.getResult().getUsers().begin().getCurrent().getOperand()->getOwner()))){
            input = dqX.getX();
            outType = RankedTensorType::get(outShape,rewriter.getI8Type());
            outShape = outType.getShape();
            quantized = true;
        }

        SmallVector<int64_t> padsOnnx = AttributeVec(pad);
        SmallVector<int64_t> stridesOnnx = AttributeVec(stride);
        SmallVector<int64_t> dilationOnnx = AttributeVec(dilation);
        SmallVector<int64_t> kernelOnnx = AttributeVec(kernel);


        // TOSA expects [top, bottom, left, right]
        auto padsTosa = rewriter.getDenseI64ArrayAttr(
            ArrayRef<int64_t>{padsOnnx[0], padsOnnx[2], padsOnnx[1], padsOnnx[3]});

        auto stridesTosa   = rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>(stridesOnnx));
        auto dilationsTosa = rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>(dilationOnnx));
        auto kernelTosa = rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>(kernelOnnx));

        RankedTensorType inputType = cast<RankedTensorType>(input.getType());
        auto inputShape = inputType.getShape();
        auto inputTosaShape = RankedTensorType::get({inputShape[0],inputShape[2],inputShape[3],inputShape[1]},inputType.getElementType());
        auto permsTosa = DenseI32ArrayAttr::get(op.getContext(),{0,2,3,1});
        input = rewriter.create<mlir::tosa::TransposeOp>(loc,inputTosaShape,input,permsTosa).getResult();

        auto maxpoolOutputShape = RankedTensorType::get({outShape[0],outShape[2],outShape[3],outShape[1]},outType.getElementType());
        Value maxpool = rewriter.create<mlir::tosa::MaxPool2dOp>(loc,maxpoolOutputShape,input,kernelTosa,stridesTosa,padsTosa).getResult();

        auto perms = DenseI32ArrayAttr::get(op.getContext(),{0,3,1,2});
        maxpool = rewriter.create<mlir::tosa::TransposeOp>(loc,outType,maxpool,perms).getResult();

        if (quantized){
            rewriter.replaceOp(qO, maxpool);
            rewriter.eraseOp(op);
            rewriter.eraseOp(dqX);
            return success();
        }
        rewriter.replaceOp(op, maxpool);
        return success();
    }
    };


} //namspace
void populateLoweringONNXMaxPoolSingleOutOpToTOSAPattern(
    ConversionTarget &target, RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
    patterns.insert<ONNXMaxPoolSingleOutOpLoweringToTosa>(typeConverter, ctx);
}
} //namespace onnx_mlir


#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include <src/Dialect/Mlir/IndexExpr.hpp>

#include <cmath>
#include <algorithm>





using namespace mlir;

namespace onnx_mlir {

namespace {

double getDoubleFromScale(Value scale) {
    auto onnxConst = cast<DenseFPElementsAttr>(scale.getDefiningOp<ONNXConstantOp>().getValueAttr());
    return onnxConst.getValues<APFloat>()[0].convertToDouble();
}


struct ONNXSigmoidOpLoweringToTOSA : public OpConversionPattern<ONNXSigmoidOp> {
    using OpConversionPattern<ONNXSigmoidOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(ONNXSigmoidOp op, typename ONNXSigmoidOp::Adaptor adaptor,ConversionPatternRewriter &rewriter) const override {


        Location loc = op.getLoc();
        Value input = adaptor.getX();

        RankedTensorType inputType = cast<RankedTensorType>(input.getType());
        auto inputshape = inputType.getShape();
        bool quantized = false;

        // Check if this is in a QDQ block
        ONNXDequantizeLinearOp dqX;
        ONNXQuantizeLinearOp qO;
        Value inScale;
        if ((dqX = dyn_cast<ONNXDequantizeLinearOp>(input.getDefiningOp())) 
          && (qO = dyn_cast<ONNXQuantizeLinearOp>(op.getResult().getUsers().begin().getCurrent().getOperand()->getOwner()))){
            input = dqX.getX();
            inputType = cast<RankedTensorType>(input.getType());
            inScale = dqX.getXScale();
            quantized = true;
        }

        // Transpose to Tosa shape [N,H,W,C]
        SmallVector<int64_t> inDims = mlir::tosa::convertFromMlirShape(inputshape);
        auto transpType = RankedTensorType::get({inDims[0],inDims[2],inDims[3],inDims[1]},inputType.getElementType());
        DenseI32ArrayAttr tosaPerms = DenseI32ArrayAttr::get(op.getContext(), {0,2,3,1});
        input = rewriter.create<mlir::tosa::TransposeOp>(loc, transpType,input,tosaPerms).getResult();
        inputType = cast<RankedTensorType>(input.getType());

        double inputScale = getDoubleFromScale(inScale);
        double minmax = 127 * inputScale;

        llvm::errs() << "\nMinmax: " << minmax << "\n";

        std::array<double,256> linspace;
        // double minmax = 4;
        int bits = 255;
        for (int i = 0; i <= bits; i++){
            double exponent = -minmax + (minmax * i * 2.0 / bits);
            linspace[i] = 1.0 / (1.0 + exp(-exponent));
            // llvm::errs() << exponent << "  ";
            llvm::errs() << linspace[i] << "  ";
        }
        double scale =  linspace[bits] / 127.0;

        llvm::errs() << "\nScale: " << scale << "\n" << "Last in linspace: " << linspace[bits] << "\n";

        SmallVector<int8_t> tableVals;
        for (int i = 0; i <= bits; i++){
            int8_t val = std::clamp(static_cast<int32_t>(round(linspace[i] / scale)), -128, 127);
            tableVals.push_back(static_cast<int8_t>(val));
            llvm::errs() << static_cast<int64_t>(val) << "  ";
        }
        
        RankedTensorType tableType = RankedTensorType::get({256},rewriter.getI8Type());
        
        DenseIntElementsAttr table = DenseIntElementsAttr::get(tableType,ArrayRef<int8_t>(tableVals));
        Value lookUpTable = rewriter.create<mlir::tosa::ConstOp>(loc,tableType,table);

        input = rewriter.create<mlir::tosa::TableOp>(loc,inputType,input,lookUpTable).getResult();

        auto top = input.getDefiningOp()->getOperand(1);
        top.dump();

        // Transpose back to ONNX shape [N,C,H,W]
        DenseI32ArrayAttr onnxPerms = DenseI32ArrayAttr::get(op.getContext(),{0,3,1,2});
        transpType = RankedTensorType::get(inputshape, inputType.getElementType());
        input = rewriter.create<mlir::tosa::TransposeOp>(loc, transpType,input,onnxPerms);


        if (quantized){
            rewriter.replaceOp(qO, input);
            rewriter.eraseOp(op);
            rewriter.eraseOp(dqX);
            return success();
        }
        rewriter.replaceOp(op, input);
        return success();
    }
};


}  //namespace


void populateLoweringONNXSigmoidOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXSigmoidOpLoweringToTOSA>(typeConverter, ctx);  // Add typeConverter
}

} //namspace onnx_mlir
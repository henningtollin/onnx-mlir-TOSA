

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace{


class ONNXCustomOpLoweringToTosa : public OpConversionPattern<ONNXCustomOp> {public:
    using OpConversionPattern<ONNXCustomOp>::OpConversionPattern;
    using OpAdaptor = typename ONNXCustomOp::Adaptor;
    LogicalResult matchAndRewrite(ONNXCustomOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        Location loc = op.getLoc();

        StringRef functionName = op.getFunctionName();

        if (functionName == "QGemm"){

        auto in = adaptor.getInputs();
        auto input = in[0];
        auto inputScale = in[1];
        auto inputZeropoint = in[2];
        auto weights = in[3];
        auto weightsScale = in[4];
        auto weightZeropoint = in[5];
        auto bias = in[6];
        auto outputScale = in[7];
        auto outputZeropoint = in[8];

        auto attr = adaptor.getAttributes();
        Attribute alpha = attr.get("alpha");
        Attribute transB = attr.get("transB");
        int64_t tB = dyn_cast<IntegerAttr>(transB).getSInt();
        

        // Transpose the weights if needed

        Value tWeights;
        if (tB == 1){
            auto weightsType = cast<RankedTensorType>(weights.getType());
            auto weightsShape = weightsType.getShape();
            auto tWShape = RankedTensorType::get({weightsShape[1], weightsShape[0]}, weightsType.getElementType());
            auto perms = DenseI32ArrayAttr::get(op.getContext(),{1,0});
            tWeights = rewriter.create<mlir::tosa::TransposeOp>(loc, tWShape, weights, perms).getResult();
        } else {
            tWeights = weights;
        }

        auto inputShape = cast<RankedTensorType>(input.getType()).getShape();
        auto tWShape = cast<RankedTensorType>(tWeights.getType()).getShape(); 
        auto outputType = RankedTensorType::get({1, inputShape[0],tWShape[1]},rewriter.getI32Type());

        auto wCast3DType = RankedTensorType::get({1, tWShape[0], tWShape[1]}, rewriter.getI8Type());
        Value castedW = rewriter.create<mlir::tosa::CastOp>(loc,wCast3DType, tWeights).getResult();

        auto iCast3DType = RankedTensorType::get({1, inputShape[0], inputShape[1]}, rewriter.getI8Type());
        Value castedInput = rewriter.create<mlir::tosa::CastOp>(loc,iCast3DType,input);

        // Cast the zeropoint for the input, weights and output to be a <1xi8> instead of <i8> as the adaptor gives
        auto zpSignlessType = RankedTensorType::get({1}, rewriter.getI8Type());
        Value castedInputZeropointMatmul = rewriter.create<mlir::tosa::CastOp>(loc, zpSignlessType, inputZeropoint).getResult();
        Value castedWeightsZeropointMatmul = rewriter.create<mlir::tosa::CastOp>(loc, zpSignlessType, weightZeropoint).getResult();
        

        //create the matmul operator

        Value matmul = rewriter.create<mlir::tosa::MatMulOp>(loc, outputType, castedInput, castedW, castedInputZeropointMatmul, castedWeightsZeropointMatmul).getResult();
        Value castedMatmul = rewriter.create<mlir::tosa::CastOp>(loc, RankedTensorType::get({inputShape[0],tWShape[1]}, rewriter.getI32Type()), matmul).getResult();

        // Cast the bias into <1x?xi32> tensor
        auto biasShape = cast<RankedTensorType>(bias.getType()).getShape();
        auto biasType = RankedTensorType::get({1,biasShape[0]},rewriter.getI32Type());
        Value castBias = rewriter.create<mlir::tosa::CastOp>(loc,biasType,bias).getResult();

        
        Value addedBias = rewriter.create<mlir::tosa::AddOp>(loc, RankedTensorType::get({inputShape[0],tWShape[1]}, rewriter.getI32Type()), castedMatmul, castBias).getResult();


        //Lambda function to get the scale values as doubles
        auto getDoubleFromScale = [&](Value scale) -> double {
          auto tosaConst = scale.getDefiningOp<mlir::tosa::ConstOp>();
          auto fpElemAttr = tosaConst.getValuesAttr();
          auto denseFpElemAttr = dyn_cast<DenseFPElementsAttr>(fpElemAttr);
          return denseFpElemAttr.getSplatValue<mlir::APFloat>().convertToDouble();
        };
        
        //Extract the scale values as doubles from the tosa.const
        double input_scale = getDoubleFromScale(inputScale);
        double weight_scale = getDoubleFromScale(weightsScale);
        double output_scale = getDoubleFromScale(outputScale);

        int32_t multiplier;
        int32_t shifts;
        double op_tensor_scale = (input_scale * weight_scale) / output_scale;
        mlir::tosa::computeMultiplierAndShift(op_tensor_scale,multiplier,shifts,16);

        auto multipliers = {multiplier};
        SmallVector<int16_t> mulvec (multipliers.begin(), multipliers.end());
        auto mulTy = RankedTensorType::get({1}, rewriter.getIntegerType(16));
        
        auto mulAttr = DenseElementsAttr::get(mulTy, ArrayRef<int16_t>(mulvec));
        Value multVal = rewriter.create<mlir::tosa::ConstOp>(loc, mulTy, mulAttr).getResult();
        
        auto shTy   = RankedTensorType::get({1}, rewriter.getIntegerType(8));
        auto shAttr = DenseElementsAttr::get(shTy, {static_cast<int8_t>(shifts)});
        Value shiftVal = rewriter.create<mlir::tosa::ConstOp>(loc, shTy, shAttr).getResult();



        bool scale32 = false;
        bool perChannel = false;
        bool inputUnsigned = false;
        bool outputUnsigned = false;
        StringRef s1("SINGLE_ROUND");

        Value castedInputZeropoint = rewriter.create<mlir::tosa::CastOp>(loc, RankedTensorType::get({1}, rewriter.getI32Type()), inputZeropoint).getResult();
        Value castedOutputZeropoint = rewriter.create<mlir::tosa::CastOp>(loc, RankedTensorType::get({1},rewriter.getI8Type()), outputZeropoint).getResult();
        
        auto outputI8Type = RankedTensorType::get({inputShape[0],tWShape[1]},rewriter.getI8Type());
        Value rescaleOutput = rewriter.create<mlir::tosa::RescaleOp>(loc,outputI8Type,addedBias,multVal,shiftVal,castedInputZeropoint,castedOutputZeropoint,scale32,s1,perChannel,inputUnsigned,outputUnsigned).getResult();


        rewriter.replaceOp(op, rescaleOutput);
        
        return success();
        } // end QGemm


        return failure();
    }
    

};


} //namespace

void populateLoweringONNXCustomOpToTOSAPattern(ConversionTarget &target, RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *ctx) {
    patterns.insert<ONNXCustomOpLoweringToTosa>(typeConverter,ctx);
}

} // namespace onnx_mlir
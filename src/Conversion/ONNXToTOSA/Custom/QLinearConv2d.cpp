
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"

using namespace mlir;
using namespace tosa;

namespace onnx_mlir {

namespace {

class ONNXQLinearConvOpLoweringToTosa : public OpConversionPattern<ONNXQLinearConvOp> { public:
    using OpConversionPattern<ONNXQLinearConvOp>::OpConversionPattern;
    using OpAdaptor = typename ONNXQLinearConvOp::Adaptor;
    LogicalResult matchAndRewrite(ONNXQLinearConvOp op, typename ONNXQLinearConvOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override {

        Location loc = op->getLoc();

        // Load all the inputs from the QLinearConv operator

        auto input = adaptor.getX();
        auto weights = adaptor.getW();
        auto bias = adaptor.getB();
        auto weightsType = cast<RankedTensorType>(weights.getType());
        auto ws = weightsType.getShape();

        
      
        auto inputScale = adaptor.getXScale();
        auto inputZeropoint = adaptor.getXZeroPoint();
        RankedTensorType inputZeropointType = dyn_cast<RankedTensorType>(inputZeropoint.getType());

        auto weightScale = adaptor.getWScale();
        auto weightZeropoint = adaptor.getWZeroPoint();
        RankedTensorType weightZeropointType = dyn_cast<RankedTensorType>(weightZeropoint.getType());

        auto outputScale = adaptor.getYScale();
        auto outputZeropoint = adaptor.getYZeroPoint();
        RankedTensorType outputZeropointType = dyn_cast<RankedTensorType>(outputZeropoint.getType());

        //Create a bias if there is none

        if (mlir::isa<NoneType>(bias.getType())) {
            DenseElementsAttr newBiasAttr = DenseElementsAttr::get(RankedTensorType::get({ws[0]}, rewriter.getI32Type()),ArrayRef<int32_t>(0));
            bias = rewriter.create<mlir::tosa::ConstOp>(loc, newBiasAttr.getType(), newBiasAttr);
        }
        
        //Lambda function to get the scale values as doubles
        auto getDoubleFromScale = [&](Value scale) -> double {
          auto tosaConst = scale.getDefiningOp<mlir::tosa::ConstOp>();
          auto fpElemAttr = tosaConst.getValuesAttr();
          auto denseFpElemAttr = dyn_cast<DenseFPElementsAttr>(fpElemAttr);
          return denseFpElemAttr.getSplatValue<mlir::APFloat>().convertToDouble();
        };
        
        //Extract the scale values as doubles from the tosa.const
        double input_scale = getDoubleFromScale(inputScale);
        double weight_scale = getDoubleFromScale(weightScale);
        double output_scale = getDoubleFromScale(outputScale);

        // Compute the multiplier and shift with a built in method found in Tosaquantutils.h
        // ## Here we can change and use our own if we want to ##
        int32_t multiplier;
        int32_t shifts;
        double op_tensor_scale = (input_scale * weight_scale) / output_scale;
        mlir::tosa::computeMultiplierAndShift(op_tensor_scale,multiplier,shifts,16);

        auto multipliers = {multiplier};
        SmallVector<int16_t> mulvec (multipliers.begin(), multipliers.end());
        auto mulTy = RankedTensorType::get({1}, rewriter.getI16Type());
        
        auto mulAttr = DenseElementsAttr::get(mulTy, ArrayRef<int16_t>(mulvec));
        Value multVal = rewriter.create<mlir::tosa::ConstOp>(loc, mulTy, mulAttr).getResult();
        
        auto shTy   = RankedTensorType::get({1}, rewriter.getI8Type());
        auto shAttr = DenseElementsAttr::get(shTy, {static_cast<int8_t>(shifts)});
        Value shiftVal = rewriter.create<mlir::tosa::ConstOp>(loc, shTy, shAttr).getResult();

        // Cast the zeropoint for the input, weights and output to be a <1xi8> instead of <i8> as the adaptor gives
        RankedTensorType out = RankedTensorType::get({1},inputZeropointType.getElementType());

        auto inputZeropointVals = cast<DenseIntElementsAttr>(inputZeropoint.getDefiningOp<mlir::tosa::ConstOp>().getValues());
        auto weightZeropointVals = cast<DenseIntElementsAttr>(weightZeropoint.getDefiningOp<mlir::tosa::ConstOp>().getValues());
        auto outputZeropointVals = cast<DenseIntElementsAttr>(outputZeropoint.getDefiningOp<mlir::tosa::ConstOp>().getValues());
        auto getZpVal = [](DenseIntElementsAttr aa) -> int64_t {
            return aa.getValues<APInt>()[0].getSExtValue();
        };
        int64_t izp = getZpVal(inputZeropointVals);
        int64_t wzp = getZpVal(weightZeropointVals);
        int64_t ozp = getZpVal(outputZeropointVals);

        auto izpCorrShape = DenseElementsAttr::get(out, ArrayRef<int8_t>(static_cast<int8_t>(izp)));
        auto wzpCorrShape = DenseElementsAttr::get(out, ArrayRef<int8_t>(static_cast<int8_t>(wzp)));
        auto ozpCorrShape = DenseElementsAttr::get(out, ArrayRef<int8_t>(static_cast<int8_t>(ozp)));
        
        Value castedInputZeropoint = rewriter.create<mlir::tosa::ConstOp>(loc, out, izpCorrShape).getResult();
        Value castedWeightsZeropoint = rewriter.create<mlir::tosa::ConstOp>(loc, out, wzpCorrShape).getResult();
        Value castedOutputZeropoint = rewriter.create<mlir::tosa::ConstOp>(loc, out, ozpCorrShape).getResult();

        // Get the types and element types for the input weights and the output.

        auto outputType = cast<RankedTensorType>(getTypeConverter()->convertType(op.getResult().getType()));
        auto outputElemType = outputType.getElementType();

        auto inputType = cast<RankedTensorType>(input.getType());
        auto inputElemType = inputType.getElementType();
        auto weightType = cast<RankedTensorType>(weights.getType());
        auto weightElemType = weightType.getElementType();

        // Load all the attributes from QLinearConv opertor

        auto pads = op.getPadsAttr();
        auto strides = op.getStridesAttr();
        auto dilations = op.getDilationsAttr();

        auto getI64 = [&](ArrayAttr a, int i, int64_t dflt) -> int64_t {
            return a ? cast<IntegerAttr>(a[i]).getInt() : dflt;
        };
      
        // ONNX attribute values (H/W order)
        int64_t sh = getI64(strides,   0, 1);
        int64_t sw = getI64(strides,   1, 1);
        int64_t dh = getI64(dilations, 0, 1);
        int64_t dw = getI64(dilations, 1, 1);
      
        // Validate/prepare padding: we only support NOTSET/VALID here (literal pads).
        SmallVector<int64_t,4> padsOnnx{0,0,0,0}; // [top, left, bottom, right]
        padsOnnx[0] = getI64(pads, 0, 0);
        padsOnnx[1] = getI64(pads, 1, 0);
        padsOnnx[2] = getI64(pads, 2, 0);
        padsOnnx[3] = getI64(pads, 3, 0);
          
        // Reorder the padding since ONNX Uses [Top, Left, Bot, Right] and Tosa Expects [Top, Bot, Left, Right]
        auto padsTosa = rewriter.getDenseI64ArrayAttr(
            ArrayRef<int64_t>{padsOnnx[0], padsOnnx[2], padsOnnx[1], padsOnnx[3]});
        auto stridesTosa   = rewriter.getDenseI64ArrayAttr({sh, sw});
        auto dilationsTosa = rewriter.getDenseI64ArrayAttr({dh, dw});

        auto context = op.getContext();

        // Convert the Tensor from [N,C,W,H] -> [N,W,H,C] according to Tosa spec

        auto permsAttr = DenseI32ArrayAttr::get(context, {0, 2, 3, 1});

        auto inputNCHW = inputType.getShape();
        auto weightsNCHW = weightType.getShape();
        
        // Create tosa::TransposeOps for both the input and the weights

        SmallVector<int64_t> inputNHWCShape = {inputNCHW[0], inputNCHW[2], inputNCHW[3], inputNCHW[1]};
        auto inputNHWCType = RankedTensorType::get(inputNHWCShape,inputElemType);
        Value inputNHWC = rewriter.create<mlir::tosa::TransposeOp>(loc,inputNHWCType,input,permsAttr).getResult();

        SmallVector<int64_t> weightsNHWCShape = {weightsNCHW[0], weightsNCHW[2], weightsNCHW[3], weightsNCHW[1]};
        auto weightsNHWCType = RankedTensorType::get(weightsNHWCShape, weightElemType);
        Value weightsNHWC = rewriter.create<mlir::tosa::TransposeOp>(loc, weightsNHWCType, weights, permsAttr).getResult();
        
        // Set the type of the accumelator -> int32 from Tosa spec
        TypeAttr accType = TypeAttr::get(rewriter.getI32Type());
        

        // Compute the convolution
        auto outputNCHW = outputType.getShape();
        
        SmallVector<int64_t> outputNHWC = {outputNCHW[0], outputNCHW[2], outputNCHW[3], outputNCHW[1]};
        auto outputNHWCType = RankedTensorType::get(outputNHWC, rewriter.getI32Type());

        Value conv2d = rewriter.create<mlir::tosa::Conv2DOp>(loc, outputNHWCType, inputNHWC, weightsNHWC, bias, castedInputZeropoint, castedWeightsZeropoint, padsTosa, stridesTosa, dilationsTosa, accType).getResult();

        // Transpose the out put back to [N,C,H,W]
        
        auto perms = mlir::DenseI32ArrayAttr::get(context,{0,3,1,2});
        Value newOutput = rewriter.create<mlir::tosa::TransposeOp>(loc,RankedTensorType::get(outputNCHW, accType.getValue()),conv2d,perms).getResult();

        //Rescale the output in the accumelators from int32 to int8
        bool scale32 = false;
        bool perChannel = false;
        bool inputUnsigned = false;
        bool outputUnsigned = false;
        StringRef s1("SINGLE_ROUND");
        
        auto newOutputType = cast<RankedTensorType>(newOutput.getType());
        auto test = RankedTensorType::get({1},newOutputType.getElementType());

        Value zp = rewriter.create<mlir::tosa::ConstOp>(loc, test, DenseElementsAttr::get(test, ArrayRef<int32_t>(static_cast<int32_t>(izp)))).getResult();
        
        Value rescaleOutput = rewriter.create<mlir::tosa::RescaleOp>(loc,outputType,newOutput,multVal,shiftVal,zp,castedOutputZeropoint,scale32,s1,perChannel,inputUnsigned,outputUnsigned).getResult();

        rewriter.replaceOp(op, rescaleOutput);

        return success();
    }
};
} // namespace

void populateLoweringONNXQLinearConvOpToTOSAPattern(ConversionTarget &target, RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *ctx) {
    patterns.insert<ONNXQLinearConvOpLoweringToTosa>(typeConverter,ctx);
}
} // namespace onnx_mlir

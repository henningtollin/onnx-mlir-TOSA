
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include <src/Dialect/Mlir/IndexExpr.hpp>

using namespace mlir;

namespace onnx_mlir {


namespace{

DenseElementsAttr castZeropoints(RankedTensorType out, Value izp, int outSize){
    auto izpv = cast<DenseIntElementsAttr>(izp.getDefiningOp<mlir::ONNXConstantOp>().getValueAttr());
    int64_t intval = izpv.getValues<APInt>()[0].getSExtValue();
    if (outSize == 8)
        return DenseElementsAttr::get(out, ArrayRef<int8_t>(static_cast<int8_t>(intval)));
    if (outSize == 32)
        return DenseElementsAttr::get(out, ArrayRef<int32_t>(static_cast<int32_t>(0)));
    return NULL;
}
    
double getDoubleFromScale(Value scale) {
    auto onnxConst = cast<DenseFPElementsAttr>(scale.getDefiningOp<ONNXConstantOp>().getValueAttr());
    return onnxConst.getValues<APFloat>()[0].convertToDouble();
}

struct ONNXGemmOpLoweringToTOSA : public OpConversionPattern<ONNXGemmOp> {
    using OpConversionPattern<ONNXGemmOp>::OpConversionPattern;
    

    LogicalResult matchAndRewrite(ONNXGemmOp op, typename ONNXGemmOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override {

        Location loc = op.getLoc();


        // Get all the Inputs and the attributes from the adaptor.
        Value A = adaptor.getA();
        Value B = adaptor.getB();
        Value C = adaptor.getC();
        FloatAttr alpha = adaptor.getAlphaAttr();
        FloatAttr beta = adaptor.getBetaAttr();

        int64_t transA = adaptor.getTransA();
        int64_t transB = adaptor.getTransB();

        bool quantized = false;
        ONNXDequantizeLinearOp dqX = NULL;
        ONNXDequantizeLinearOp dqW = NULL;
        ONNXQuantizeLinearOp qO = NULL;
        auto AA = cast<RankedTensorType>(A.getType());
        auto tempZp = RankedTensorType::get({1},rewriter.getI8Type());
        auto tempZpVals = DenseElementsAttr::get(tempZp,ArrayRef<int8_t>((0)));
        Value inputZeropoint = rewriter.create<mlir::tosa::ConstOp>(loc,tempZp,tempZpVals).getResult();
        Value weightZeropoint = rewriter.create<mlir::tosa::ConstOp>(loc,tempZp,tempZpVals).getResult();
        Value outputZeropoint = NULL;
        Value multVal;
        Value shiftVal;
        
        double si;
        double sw;
        double so;

        // Check if this is in a QDQ block. If it is create the multiplier and shift for the rescale operator.
        if ((dqX = dyn_cast<ONNXDequantizeLinearOp>(A.getDefiningOp())) 
          && (dqW = dyn_cast<ONNXDequantizeLinearOp>(B.getDefiningOp()))
          && (qO = dyn_cast<ONNXQuantizeLinearOp>(op.getResult().getUsers().begin().getCurrent().getOperand()->getOwner()))){
            A = dqX.getX();
            B = dqW.getX();

            inputZeropoint = dqX.getXZeroPoint();
            weightZeropoint = dqW.getXZeroPoint();
            outputZeropoint = qO.getYZeroPoint();
            si = getDoubleFromScale(dqX.getXScale());
            sw = getDoubleFromScale(dqW.getXScale());
            so = getDoubleFromScale(qO.getYScale());

            auto inputZeropointType = cast<RankedTensorType>(inputZeropoint.getType());
            RankedTensorType out = RankedTensorType::get({1}, inputZeropointType.getElementType());

            DenseElementsAttr inputZeropointCast = castZeropoints(out,inputZeropoint,8);
            DenseElementsAttr weightZeropointCast = castZeropoints(out, weightZeropoint, 8);
            inputZeropoint = rewriter.create<mlir::tosa::ConstOp>(loc,out,inputZeropointCast).getResult();
            weightZeropoint = rewriter.create<mlir::tosa::ConstOp>(loc,out,weightZeropointCast).getResult();

            int32_t multiplier;
            int32_t shifts;
            double op_tensor_scale = (si * sw) / so;
            mlir::tosa::computeMultiplierAndShift(op_tensor_scale,multiplier,shifts,16);

            auto multipliers = {multiplier};
            SmallVector<int16_t> mulvec (multipliers.begin(), multipliers.end());
            auto mulTy = RankedTensorType::get({1}, rewriter.getI16Type());
            
            auto mulAttr = DenseElementsAttr::get(mulTy, ArrayRef<int16_t>(mulvec));
            multVal = rewriter.create<mlir::tosa::ConstOp>(loc, mulTy, mulAttr).getResult();
            
            auto shTy   = RankedTensorType::get({1}, rewriter.getI8Type());
            auto shAttr = DenseElementsAttr::get(shTy, {static_cast<int8_t>(shifts)});
            shiftVal = rewriter.create<mlir::tosa::ConstOp>(loc, shTy, shAttr).getResult();
            
            quantized = true;
        }

        auto AType = cast<RankedTensorType>(A.getType());
        auto BType = cast<RankedTensorType>(B.getType());
        auto AShape = AType.getShape();
        auto BShape = BType.getShape();

        // If this is a QDQ block get the bias directly, witout dequantizing it.
        if (quantized) {
            ONNXDequantizeLinearOp deQuantBias = op.getC().getDefiningOp<ONNXDequantizeLinearOp>();
            C = deQuantBias.getX();
            rewriter.eraseOp(deQuantBias);
        }
        
        // If any of the matrices A or B is known at compile time it is possible to transpose them at compile time.
        if (transA){
            RankedTensorType transAType = RankedTensorType::get(ArrayRef<int64_t>({AShape[1],AShape[0]}),AType.getElementType());
            DenseI32ArrayAttr perms = DenseI32ArrayAttr::get(op.getContext(),{1,0});
            A = rewriter.create<mlir::tosa::TransposeOp>(loc, transAType,A,perms).getResult();
            AType = cast<RankedTensorType>(A.getType());
            AShape = AType.getShape();
        }
        
        if (transB){
            // If B is a constant we can traspose it during compile time
            mlir::ONNXConstantOp defB;
            if (((defB = dyn_cast<mlir::ONNXConstantOp>(B.getDefiningOp())) != NULL) && quantized){
                auto weightElements = cast<DenseIntElementsAttr>(defB.getValueAttr());
                auto elemtTypes = weightElements.getType();
                auto elemShapes = elemtTypes.getShape();
                auto values = weightElements.getValues<int8_t>();
                SmallVector<int64_t> dimvals(elemShapes.begin(), elemShapes.end());
                int64_t H = dimvals[0], W = dimvals[1];
                SmallVector<int8_t> neworder;
                for (int j = 0; j < W; j++){
                    for (int i = 0; i < H; i++){
                        int fromind = i * W + j;
                        neworder.push_back(values[fromind]);
                    }
                }
                auto newshapesW = RankedTensorType::get(ArrayRef<int64_t>({W,H}),elemtTypes.getElementType());
                auto newWeights = DenseElementsAttr::get(newshapesW, ArrayRef<int8_t>(neworder));
                B = rewriter.create<mlir::tosa::ConstOp>(loc, newshapesW, newWeights).getResult();
            } else {
                RankedTensorType transBType = RankedTensorType::get(ArrayRef<int64_t>({BShape[1],BShape[0]}),BType.getElementType());
                DenseI32ArrayAttr perms = DenseI32ArrayAttr::get(op.getContext(),{1,0});
                B = rewriter.create<mlir::tosa::TransposeOp>(loc, transBType,B,perms).getResult();
            }
            BType = cast<RankedTensorType>(B.getType());
            BShape = BType.getShape();
        }


        // In TOSA Matmul needs the input Tensors to be rank 3. In onnx they are rank 2. Here a extra dimesion is added.
        Value ATosaShape = mlir::tosa::getTosaConstShape(rewriter,loc,{1,AShape[0],AShape[1]});
        Value BTosaShape = mlir::tosa::getTosaConstShape(rewriter, loc, {1,BShape[0],BShape[1]});

        RankedTensorType Areshaped = RankedTensorType::get({1,AShape[0],AShape[1]},AType.getElementType());
        RankedTensorType Breshaped = RankedTensorType::get({1,BShape[0],BShape[1]},BType.getElementType());
        A = rewriter.create<mlir::tosa::ReshapeOp>(loc,Areshaped,A,ATosaShape).getResult();
        B = rewriter.create<mlir::tosa::ReshapeOp>(loc,Breshaped,B,BTosaShape).getResult();

        RankedTensorType matmulshape = RankedTensorType::get({1,AShape[0],BShape[1]},rewriter.getI32Type());
        Value output = rewriter.create<mlir::tosa::MatMulOp>(loc,matmulshape,A,B,inputZeropoint,weightZeropoint);


        
        auto outputType = cast<RankedTensorType>(output.getType());
        auto outputShape = outputType.getShape();

        Value outputOnnxShape = mlir::tosa::getTosaConstShape(rewriter, loc, {outputShape[1],outputShape[2]});
        RankedTensorType onnxShape = RankedTensorType::get({outputShape[1],outputShape[2]},cast<RankedTensorType>(output.getType()).getElementType());
        output = rewriter.create<mlir::tosa::ReshapeOp>(loc,onnxShape,output,outputOnnxShape).getResult();

        // If there exist a bias create it as a constant op and then add it to the matmul.
        if(!mlir::isa<NoneType>(C.getType())){
            RankedTensorType CType = cast<RankedTensorType>(C.getType());
            auto CShape = CType.getShape();
            auto Cvals = cast<DenseElementsAttr>(C.getDefiningOp<ONNXConstantOp>().getValueAttr());
            SmallVector<int32_t> Cints;
            for (APInt i : Cvals.getValues<APInt>()){
                Cints.push_back(i.getSExtValue());
            }
            auto aa = CShape.begin();
            CType = RankedTensorType::get({1, *aa},CType.getElementType());
            auto reshapedC = DenseElementsAttr::get(CType,ArrayRef<int32_t>(Cints));
            C = rewriter.create<mlir::tosa::ConstOp>(loc,CType,reshapedC).getResult();
            output = rewriter.create<mlir::tosa::AddOp>(loc,cast<RankedTensorType>(output.getType()),output,C).getResult();
        }

        // If the this is in a quantized model rescale it to int8. And remove the quantize and dequantize ops.
        if (quantized){
            bool scale32 = false;
            bool perChannel = false;
            bool inputUnsigned = false;
            bool outputUnsigned = false;
            StringRef s1("SINGLE_ROUND");
            auto finalshape = cast<RankedTensorType>(output.getType()).getShape();
            RankedTensorType onnxShapei8 = RankedTensorType::get(finalshape,rewriter.getI8Type());
            RankedTensorType ri32type = RankedTensorType::get({1},rewriter.getI32Type());
            RankedTensorType ri8type = RankedTensorType::get({1},rewriter.getI8Type());
            DenseElementsAttr inzp = DenseElementsAttr::get(ri32type,ArrayRef<int32_t>({0}));
            DenseElementsAttr ouzp = DenseElementsAttr::get(ri8type, ArrayRef<int8_t>({0}));
            Value reinzp = rewriter.create<mlir::tosa::ConstOp>(loc,ri32type,inzp).getResult();
            Value reouzp = rewriter.create<mlir::tosa::ConstOp>(loc,ri8type,ouzp).getResult();
            Value rescaled = rewriter.create<mlir::tosa::RescaleOp>(loc,onnxShapei8,output,multVal,shiftVal,reinzp,reouzp, scale32,s1,perChannel,inputUnsigned,outputUnsigned).getResult();
            rewriter.replaceOp(qO,rescaled);
            rewriter.eraseOp(op);
            rewriter.eraseOp(dqX);
            if (dqW.getResult().use_empty())
                rewriter.eraseOp(dqW);
            return success();
        }
        rewriter.replaceOp(op, output);

        return success();
    }

};



} // namespace
void populateLoweringONNXGemmOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXGemmOpLoweringToTOSA>(typeConverter, ctx);  // Add typeConverter
}
} // namespace onnx_mlir
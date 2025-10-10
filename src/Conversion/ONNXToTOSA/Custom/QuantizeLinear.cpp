

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

namespace onnx_mlir {

namespace{

void ChangeFuncType(func::FuncOp func, ConversionPatternRewriter &rewriter){

    
   
    auto oldtype = func.getFunctionType();
    
    auto in = llvm::to_vector<4>(oldtype.getInputs());
    auto out = llvm::to_vector<4>(oldtype.getResults());
    auto inType = cast<RankedTensorType>(in[0]);
    auto outType = cast<RankedTensorType>(out[0]);
    auto inShape = inType.getShape();
    auto outShape = outType.getShape();

    auto newinput = RankedTensorType::get(inShape,rewriter.getI8Type());
    auto newoutput = RankedTensorType::get(outShape,rewriter.getI8Type());

    in[0] = newinput;
    out[0] = newoutput;
    auto newFuncType = mlir::FunctionType::get(func.getContext(), in, out);
    func.setType(newFuncType);
    Block &entry = func.front();
    BlockArgument arg = entry.getArgument(0);
    arg.setType(newinput);
}

SmallVector<int64_t> AttributeVec(ArrayAttr attr){
    
    SmallVector<int64_t> att; 
    for (size_t i = 0; i < attr.size(); i++){
        int64_t a = cast<IntegerAttr>(attr[i]).getInt();
        att.push_back(a);
    }
    return att; 
}

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
    

class ONNXQuantizeLinearOpLoweringToTosa : public OpConversionPattern<ONNXQuantizeLinearOp> { public:
    using OpConversionPattern<ONNXQuantizeLinearOp>::OpConversionPattern;
    using OpAdaptor = typename ONNXQuantizeLinearOp::Adaptor;
    LogicalResult matchAndRewrite(ONNXQuantizeLinearOp op, typename ONNXQuantizeLinearOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override {   

        Location loc = op.getLoc();
        auto func = op->getParentOfType<mlir::func::FuncOp>();
        ONNXDequantizeLinearOp inpp;
        ONNXConvOp conv;
        auto uses = op.getResult().getUsers();
        for (Operation *u : uses){
            if (auto dq = dyn_cast<ONNXDequantizeLinearOp>(u)){
                inpp = dq;
                break;
            }
        }
        if(inpp)
        for (Operation *c : inpp.getResult().getUsers()){
            if(auto convop = dyn_cast<ONNXConvOp>(c)){
                conv = convop;
                break;
            }
        }
        if(conv){
            //auto quanti = op.getX();
            auto inputZp = op.getYZeroPoint();
            auto inputScale = op.getYScale();
            ONNXDequantizeLinearOp deQuantWeight = conv.getW().getDefiningOp<ONNXDequantizeLinearOp>();
            auto quantw = deQuantWeight.getX();
            auto weightScale = deQuantWeight.getXScale();
            auto weightZp = deQuantWeight.getXZeroPoint();

            auto weightType = cast<RankedTensorType>(quantw.getType());
            auto ws = weightType.getShape();
            //If bias is zero create it
            auto biasType = conv.getB().getType();
            Value bias;
            if (mlir::isa<NoneType>(biasType)){
                DenseElementsAttr newBiasAttr = DenseElementsAttr::get(RankedTensorType::get({ws[0]}, rewriter.getI32Type()),ArrayRef<int32_t>(0));
                bias = rewriter.create<mlir::tosa::ConstOp>(loc, newBiasAttr.getType(), newBiasAttr);
            } else {
                ONNXDequantizeLinearOp deQuantBias = conv.getB().getDefiningOp<ONNXDequantizeLinearOp>();
                bias = deQuantBias.getX();
                rewriter.eraseOp(deQuantBias);
            }
            
            op->getBlock()->getArgument(0).getType();
            auto defop = op.getX().getDefiningOp();
            Value quanti;
            if (!defop){
                ChangeFuncType(func, rewriter);
                quanti = func.front().getArgument(0);
            } else {
                quanti = op.getX();
            }

            auto pads = conv.getPadsAttr();
            auto stride = conv.getStridesAttr();
            auto dilation = conv.getDilationsAttr();

            SmallVector<int64_t> padsOnnx = AttributeVec(pads);
            SmallVector<int64_t> stridesOnnx = AttributeVec(stride);
            SmallVector<int64_t> dilationOnnx = AttributeVec(dilation);
            auto padsTosa = rewriter.getDenseI64ArrayAttr(
                ArrayRef<int64_t>{padsOnnx[0], padsOnnx[2], padsOnnx[1], padsOnnx[3]});
            auto stridesTosa   = rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>(stridesOnnx));
            auto dilationsTosa = rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>(dilationOnnx));
            
            auto inputZeropointType = cast<RankedTensorType>(inputZp.getType());
            RankedTensorType out = RankedTensorType::get({1},inputZeropointType.getElementType());
            DenseElementsAttr inputZpCorrShape = castZeropoints(out, inputZp,8);
            DenseElementsAttr weightZpCorrShape = castZeropoints(out, weightZp,8);
            Value inZp = rewriter.create<mlir::tosa::ConstOp>(loc,out,inputZpCorrShape);
            Value wZp = rewriter.create<mlir::tosa::ConstOp>(loc,out,weightZpCorrShape);

            TypeAttr accType = TypeAttr::get(rewriter.getI32Type());
            auto outputType = cast<RankedTensorType>(getTypeConverter()->convertType(conv.getResult().getType()));
            auto outputShape = outputType.getShape();
            SmallVector<int64_t> outputNHWC ={outputShape[0],outputShape[2],outputShape[3],outputShape[1]};
            auto outputElemType = outputType.getElementType();
            
            RankedTensorType intOutputType = RankedTensorType::get(outputNHWC,rewriter.getI32Type());
            auto permsAttr = DenseI32ArrayAttr::get(op.getContext(), {0, 2, 3, 1});
            auto inputType = cast<RankedTensorType>(quanti.getType());
            auto inputNCHW = inputType.getShape();
            auto inputElemType = inputType.getElementType();
            
            auto weightElemType = weightType.getElementType();
            auto weightsNCHW = weightType.getShape();
            
            // Create tosa::TransposeOps for both the input and the weights
            
            SmallVector<int64_t> inputNHWCShape = {inputNCHW[0], inputNCHW[2], inputNCHW[3], inputNCHW[1]};
            auto inputNHWCType = RankedTensorType::get(inputNHWCShape,rewriter.getI8Type());
            Value inputNHWC = rewriter.create<mlir::tosa::TransposeOp>(loc,inputNHWCType,quanti,permsAttr).getResult();

            SmallVector<int64_t> weightsNHWCShape = {weightsNCHW[0], weightsNCHW[2], weightsNCHW[3], weightsNCHW[1]};
            auto weightsNHWCType = RankedTensorType::get(weightsNHWCShape, weightElemType);
            Value weightsNHWC = rewriter.create<mlir::tosa::TransposeOp>(loc, weightsNHWCType, quantw, permsAttr).getResult();
            
            
            Value tosaConv = rewriter.create<mlir::tosa::Conv2DOp>(loc,intOutputType,inputNHWC,weightsNHWC,bias
                ,inZp,wZp,padsTosa,stridesTosa,dilationsTosa,accType).getResult();

            auto convType = cast<RankedTensorType>(tosaConv.getType());
            auto convShape = convType.getShape();
            SmallVector<int64_t> outNCHW = {convShape[0], convShape[3], convShape[1], convShape[2]};
            auto perms = DenseI32ArrayAttr::get(op.getContext(), {0, 3, 1, 2});
            auto outputconvType = RankedTensorType::get(outNCHW,convType.getElementType());


            Value reshapeConv = rewriter.create<mlir::tosa::TransposeOp>(loc,outputconvType,tosaConv,perms).getResult();
            ONNXQuantizeLinearOp nextquant;
            for (Operation *nq : conv.getResult().getUsers()){
                if (nextquant = dyn_cast<ONNXQuantizeLinearOp>(nq)) break;
            }

            Value outputScale = nextquant.getYScale();
            Value outputzeropoint = nextquant.getYZeroPoint();

            RankedTensorType outi8 = RankedTensorType::get({1},rewriter.getI8Type());
            DenseElementsAttr outputZpCorrShape = castZeropoints(outi8, outputzeropoint,8);
            Value oz = rewriter.create<mlir::tosa::ConstOp>(loc,outi8,outputZpCorrShape).getResult();
            

            RankedTensorType outi32 = RankedTensorType::get({1},rewriter.getI32Type());
            DenseElementsAttr rescaleZp = castZeropoints(outi32,outputzeropoint,32);
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


            bool scale32 = false;
            bool perChannel = false;
            bool inputUnsigned = false;
            bool outputUnsigned = false;
            StringRef s1("SINGLE_ROUND");

            auto rescaledOutType = cast<RankedTensorType>(reshapeConv.getType());
            auto rescaledOutShape = rescaledOutType.getShape();
            auto outInt8 = RankedTensorType::get(rescaledOutShape,rewriter.getI8Type());

            Value zp = rewriter.create<mlir::tosa::ConstOp>(loc,outi32, rescaleZp).getResult();

            
            Value rescaleOutput = rewriter.create<mlir::tosa::RescaleOp>(loc,outInt8,reshapeConv,multVal,shiftVal,zp,oz,scale32,s1,perChannel,inputUnsigned,outputUnsigned).getResult();

            rewriter.setInsertionPointToEnd(&func.front());
            auto ret = cast<mlir::func::ReturnOp>(func.front().getTerminator());
            ret->setOperand(0, rescaleOutput);

            rewriter.replaceOp(conv,rescaleOutput);
            // rewriter.eraseOp(inpp);
            // rewriter.eraseOp(op);
            // rewriter.eraseOp(deQuantWeight);

            
            
        }

        // if(inpp && !conv){
        //     if(inpp.getResult().getUsers().empty()){
        //         rewriter.eraseOp(inpp);
        //         rewriter.eraseOp(op);
        //     }
        // }        
        return success();

    }
};
}
void populateLoweringONNXQuantizeLinearOpToTOSAPattern(ConversionTarget &target, RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *ctx) {
    patterns.insert<ONNXQuantizeLinearOpLoweringToTosa>(typeConverter,ctx);
}

}
/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Conv2D.cpp - Conv2D Op ------------------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX conv operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

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

namespace {


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

struct ONNXConvOpLoweringToTOSA : public OpConversionPattern<ONNXConvOp> {
    using OpConversionPattern<ONNXConvOp>::OpConversionPattern;
  
    LogicalResult matchAndRewrite(ONNXConvOp op,
                                  typename ONNXConvOp::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
      Location loc = op.getLoc();
  
      // The output shape fomr onnx and the type of elements in the output
      auto expectedNCHW = cast<RankedTensorType>(
          getTypeConverter()->convertType(op.getResult().getType()));
      auto elemTy = expectedNCHW.getElementType();
  
      // Read the inputs from the adaptor to translate them into Tosa directly.
      Value X = adaptor.getX();
      Value W = adaptor.getW();
      Value B = adaptor.getB();
      ONNXDequantizeLinearOp dqX = NULL;
      ONNXDequantizeLinearOp dqW = NULL;
      ONNXQuantizeLinearOp qO = NULL;

      Value inputZeropoint = NULL;
      Value weightZeropoint = NULL;
      Value outputZeropoint = NULL;
      Value multVal;
      Value shiftVal;

      double si;
      double sw;
      double so;


      bool quantized = false;

      // Check it the opertors before and after is Quantize and DeQuantize. If they a rescale is needed after the Conv operator
      // and mult and shift is calculated.
      if ((dqX = dyn_cast<ONNXDequantizeLinearOp>(X.getDefiningOp())) 
          && (dqW = dyn_cast<ONNXDequantizeLinearOp>(W.getDefiningOp()))
          && (qO = dyn_cast<ONNXQuantizeLinearOp>(op.getResult().getUsers().begin().getCurrent().getOperand()->getOwner()))){
        X = dqX.getX();
        W = dqW.getX();

        // Get the scales and zeropoints from the surrounding qunatize and dequantize operators.
        inputZeropoint = dqX.getXZeroPoint();
        weightZeropoint = dqW.getXZeroPoint();
        outputZeropoint = qO.getYZeroPoint();
        si = getDoubleFromScale(dqX.getXScale());
        sw = getDoubleFromScale(dqW.getXScale());
        so = getDoubleFromScale(qO.getYScale());

        auto inputZeropointType = cast<RankedTensorType>(inputZeropoint.getType());
        RankedTensorType out = RankedTensorType::get({1}, inputZeropointType.getElementType());

        // Give the zeropoints the proper shape and element type.
        DenseElementsAttr inputZeropointCast = castZeropoints(out,inputZeropoint,8);
        DenseElementsAttr weightZeropointCast = castZeropoints(out, weightZeropoint, 8);
        inputZeropoint = rewriter.create<mlir::tosa::ConstOp>(loc,out,inputZeropointCast).getResult();
        weightZeropoint = rewriter.create<mlir::tosa::ConstOp>(loc,out,weightZeropointCast).getResult();
          
        // Create the multiplier and shift for the rescale
        int32_t multiplier;
        int32_t shifts;
        double op_tensor_scale = (si * sw) / so;
        mlir::tosa::computeMultiplierAndShift(op_tensor_scale,multiplier,shifts,16);

        auto multipliers = {multiplier};
        SmallVector<int16_t> mulvec (multipliers.begin(), multipliers.end());
        auto mulTy = RankedTensorType::get({1}, rewriter.getI16Type());
        
        // Create the const op for the muliplier and the shift value.
        auto mulAttr = DenseElementsAttr::get(mulTy, ArrayRef<int16_t>(mulvec));
        multVal = rewriter.create<mlir::tosa::ConstOp>(loc, mulTy, mulAttr).getResult();
        
        auto shTy   = RankedTensorType::get({1}, rewriter.getI8Type());
        auto shAttr = DenseElementsAttr::get(shTy, {static_cast<int8_t>(shifts)});
        shiftVal = rewriter.create<mlir::tosa::ConstOp>(loc, shTy, shAttr).getResult();

        quantized = true;
      } 

      auto weightType = cast<RankedTensorType>(W.getType());
      auto ws = weightType.getShape();

      // If no bias is provided a new bias vector of zeros is created. If we are in a quantized model the bias is taken before it gets dequantized again.
      if (mlir::isa<NoneType>(B.getType())){
        DenseElementsAttr newBiasAttr = DenseElementsAttr::get(RankedTensorType::get({ws[0]}, rewriter.getI32Type()),ArrayRef<int32_t>(0));
        B = rewriter.create<mlir::tosa::ConstOp>(loc, newBiasAttr.getType(), newBiasAttr);
      } else if (quantized) {
        ONNXDequantizeLinearOp deQuantBias = op.getB().getDefiningOp<ONNXDequantizeLinearOp>();
        B = deQuantBias.getX();
        rewriter.eraseOp(deQuantBias);
      }

      // Get the attributes from the the ONNX version of conv2d
      auto stride   = op.getStridesAttr();   
      auto dilation = op.getDilationsAttr(); 
      auto pad      = op.getPadsAttr();      
      auto autoPadAttr   = op.getAutoPadAttr();   
      int64_t group      = adaptor.getGroup();
  
      // Get the attributes as smallvectors such that we can easily use them later
      SmallVector<int64_t> padsOnnx = AttributeVec(pad);
      SmallVector<int64_t> stridesOnnx = AttributeVec(stride);
      SmallVector<int64_t> dilationOnnx = AttributeVec(dilation);

      // TOSA expects [top, bottom, left, right] which is not the same order as we get them from ONNX
      auto padsTosa = rewriter.getDenseI64ArrayAttr(
          ArrayRef<int64_t>{padsOnnx[0], padsOnnx[2], padsOnnx[1], padsOnnx[3]});

      // Create the attributes for Tosa.    
      auto stridesTosa   = rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>(stridesOnnx));
      auto dilationsTosa = rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>(dilationOnnx));

      // Transpose the input tensor such the it fits the Tosa standard [N,H,W,C]
      auto inNCHW = cast<RankedTensorType>(X.getType()).getShape(); 
      SmallVector<int64_t,4> xNHWCShape{inNCHW[0], inNCHW[2], inNCHW[3], inNCHW[1]};
      auto xNHWCTy = RankedTensorType::get(xNHWCShape, cast<RankedTensorType>(X.getType()).getElementType());
      auto pN2HWC  = DenseI32ArrayAttr::get(op.getContext(), {0,2,3,1});
      Value xNHWC  = rewriter.create<mlir::tosa::TransposeOp>( loc, xNHWCTy, X, pN2HWC).getResult();
  
      // The weights also has to be transposed into Tosa standard
      auto wOICK = cast<RankedTensorType>(W.getType()).getShape();
      

      // This if will transpose the weights for the conv at compile time and then the transpose can be skipped.
      Value NewTosaConst;
      if (quantized){
        mlir::ONNXConstantOp wop = W.getDefiningOp<mlir::ONNXConstantOp>();
        auto weightElements = cast<DenseIntElementsAttr>(wop.getValueAttr());
        auto elemtTypes = weightElements.getType();
        auto elemShapes = elemtTypes.getShape();
        
        auto values = weightElements.getValues<int8_t>();
        SmallVector<int64_t> dimvals(elemShapes.begin(), elemShapes.end());
        int64_t N = dimvals[0], C = dimvals[1], H = dimvals[2], W = dimvals[3];
        SmallVector<int8_t> neworder;
        for (int i = 0; i < N; i++){
          for (int k = 0; k < H; k++){
            for (int l = 0; l < W; l++){
              for (int j = 0; j < C; j++){
                int fromind = i * C * H * W + j * H * W + k * W + l;
                neworder.push_back(values[fromind]);
              }
            }
          }
        }
        auto newshapesW = RankedTensorType::get(ArrayRef<int64_t>({N,H,W,C}),elemtTypes.getElementType());
        auto newWeights = DenseElementsAttr::get(newshapesW, ArrayRef<int8_t>(neworder));
        NewTosaConst = rewriter.create<mlir::tosa::ConstOp>(loc, newshapesW, newWeights).getResult();
      }

      
      
      SmallVector<int64_t,4> wOKWIC{wOICK[0], wOICK[2], wOICK[3], wOICK[1]};
      auto wOKWICTy = RankedTensorType::get(wOKWIC,
          cast<RankedTensorType>(W.getType()).getElementType());
      auto pW   = DenseI32ArrayAttr::get(op.getContext(), {0,2,3,1});
      Value wOKWICv = rewriter.create<mlir::tosa::TransposeOp>(loc, wOKWICTy, W, pW).getResult();

      // Set the type for the accumerlator of the Conv. Dependant on what the ONNX model has.
      TypeAttr accType = TypeAttr::get(elemTy.isF16() ? rewriter.getF16Type()
                                                      : rewriter.getF32Type());

      
      // Create the shape for the output and use it to transpose the conv back to ONNX shape.
      auto outNCHW = expectedNCHW.getShape();
      SmallVector<int64_t,4> outNHWC{outNCHW[0], outNCHW[2], outNCHW[3], outNCHW[1]};
      auto convOutNHWC = RankedTensorType::get(outNHWC, elemTy);
  
      Value yNHWC;
      Value yNCHW;
      auto pH2NCW = DenseI32ArrayAttr::get(op.getContext(), {0,3,1,2});
      
      if (!quantized){
        yNHWC = rewriter.create<mlir::tosa::Conv2DOp>(loc, convOutNHWC, xNHWC, wOKWICv, B,padsTosa, stridesTosa, dilationsTosa, accType).getResult();
        yNCHW = rewriter.create<mlir::tosa::TransposeOp>(loc, expectedNCHW, yNHWC, pH2NCW).getResult();
      } else {
        // This is for the QDQ version. It is important to create the rescale before the transpose as we might later want to remove the transpose. This has to do with the shape of the output.
        // Here the conv is made and then the output is rescaled into int8 and finally transposed back to ONNX shape.
        accType = TypeAttr::get(rewriter.getI32Type());
        yNHWC = rewriter.create<mlir::tosa::Conv2DOp>(loc,RankedTensorType::get(outNHWC, rewriter.getI32Type()), xNHWC, NewTosaConst, B, inputZeropoint, weightZeropoint, padsTosa, stridesTosa, dilationsTosa, accType).getResult();
        auto expectedNHWC = RankedTensorType::get(convOutNHWC.getShape(),rewriter.getI8Type());
        bool scale32 = false;
        bool perChannel = false;
        bool inputUnsigned = false;
        bool outputUnsigned = false;
        StringRef s1("SINGLE_ROUND");

        RankedTensorType ri32type = RankedTensorType::get({1},rewriter.getI32Type());
        RankedTensorType ri8type = RankedTensorType::get({1},rewriter.getI8Type());
        // The zeropoint is always set to zero since this lowering is only for scale quantization as of now. 
        DenseElementsAttr inzp = DenseElementsAttr::get(ri32type,ArrayRef<int32_t>({0}));
        DenseElementsAttr ouzp = DenseElementsAttr::get(ri8type, ArrayRef<int8_t>({0}));
        Value reinzp = rewriter.create<mlir::tosa::ConstOp>(loc,ri32type,inzp).getResult();
        Value reouzp = rewriter.create<mlir::tosa::ConstOp>(loc,ri8type,ouzp).getResult();
        Value rescaled = rewriter.create<mlir::tosa::RescaleOp>(loc,expectedNHWC,yNHWC,multVal,shiftVal,reinzp,reouzp, scale32,s1,perChannel,inputUnsigned,outputUnsigned).getResult();
        
        yNCHW = rewriter.create<mlir::tosa::TransposeOp>(loc, RankedTensorType::get(outNCHW,rewriter.getI8Type()), rescaled, pH2NCW).getResult();


        // Replace the operator
        rewriter.replaceOp(qO,yNCHW);
        rewriter.eraseOp(op);
        rewriter.eraseOp(dqX);
        if (dqW.getResult().use_empty())
          rewriter.eraseOp(dqW);
        return success();
      }
  
      
      // Replace the operator
      rewriter.replaceOp(op, yNCHW);
      return success();
    }
  };
  
} // namespace

void populateLoweringONNXConvOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXConvOpLoweringToTOSA>(typeConverter, ctx);  // Add typeConverter
}

} // namespace onnx_mlir
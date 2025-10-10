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
  
      // ---- 0) Expected result type (what the conversion framework wants) ----
      auto expectedNCHW = cast<RankedTensorType>(
          getTypeConverter()->convertType(op.getResult().getType()));
      auto elemTy = expectedNCHW.getElementType();
  
      // ---- 1) Read operands (converted) ----
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


      if ((dqX = dyn_cast<ONNXDequantizeLinearOp>(X.getDefiningOp())) 
          && (dqW = dyn_cast<ONNXDequantizeLinearOp>(W.getDefiningOp()))
          && (qO = dyn_cast<ONNXQuantizeLinearOp>(op.getResult().getUsers().begin().getCurrent().getOperand()->getOwner()))){
        X = dqX.getX();
        W = dqW.getX();
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

      auto weightType = cast<RankedTensorType>(W.getType());
      auto ws = weightType.getShape();

      if (mlir::isa<NoneType>(B.getType())){
        DenseElementsAttr newBiasAttr = DenseElementsAttr::get(RankedTensorType::get({ws[0]}, rewriter.getI32Type()),ArrayRef<int32_t>(0));
        B = rewriter.create<mlir::tosa::ConstOp>(loc, newBiasAttr.getType(), newBiasAttr);
      } else if (quantized) {
        ONNXDequantizeLinearOp deQuantBias = op.getB().getDefiningOp<ONNXDequantizeLinearOp>();
        B = deQuantBias.getX();
        rewriter.eraseOp(deQuantBias);
      }
  
      // ---- 2) Read attributes directly from the ONNX op ----
      // Defaults per ONNX spec: strides/dilations default to [1,1] if absent.
      auto stride   = op.getStridesAttr();   // ArrayAttr or null
      auto dilation = op.getDilationsAttr(); // ArrayAttr or null
      auto pad      = op.getPadsAttr();      // ArrayAttr or null
      auto autoPadAttr   = op.getAutoPadAttr();   // StringAttr ("NOTSET", "VALID", "SAME_*")
      int64_t group      = adaptor.getGroup();
  
      SmallVector<int64_t> padsOnnx = AttributeVec(pad);
      SmallVector<int64_t> stridesOnnx = AttributeVec(stride);
      SmallVector<int64_t> dilationOnnx = AttributeVec(dilation);

      // TOSA expects [top, bottom, left, right]
      auto padsTosa = rewriter.getDenseI64ArrayAttr(
          ArrayRef<int64_t>{padsOnnx[0], padsOnnx[2], padsOnnx[1], padsOnnx[3]});

      auto stridesTosa   = rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>(stridesOnnx));
      auto dilationsTosa = rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>(dilationOnnx));
  
      // ---- 3) Layout bridges (type them explicitly) ----
      auto inNCHW = cast<RankedTensorType>(X.getType()).getShape(); // [N,C,H,W] (N/H/W may be ?)
      SmallVector<int64_t,4> xNHWCShape{inNCHW[0], inNCHW[2], inNCHW[3], inNCHW[1]};
      auto xNHWCTy = RankedTensorType::get(xNHWCShape, cast<RankedTensorType>(X.getType()).getElementType());
      auto pN2HWC  = DenseI32ArrayAttr::get(op.getContext(), {0,2,3,1});
      Value xNHWC  = rewriter.create<mlir::tosa::TransposeOp>( loc, xNHWCTy, X, pN2HWC).getResult();
  
      // Weights [OC,IC,KH,KW] → [OC,KH,KW,IC]
      auto wOICK = cast<RankedTensorType>(W.getType()).getShape();
      if (ShapedType::isDynamic(wOICK[0]) || ShapedType::isDynamic(wOICK[1]) ||
          ShapedType::isDynamic(wOICK[2]) || ShapedType::isDynamic(wOICK[3]))
        return rewriter.notifyMatchFailure(op, "weight shape must be static");
      int64_t OC = wOICK[0];
      SmallVector<int64_t,4> wOKWIC{wOICK[0], wOICK[2], wOICK[3], wOICK[1]};
      auto wOKWICTy = RankedTensorType::get(wOKWIC,
          cast<RankedTensorType>(W.getType()).getElementType());
      auto pW   = DenseI32ArrayAttr::get(op.getContext(), {0,2,3,1});
      Value wOKWICv = rewriter.create<mlir::tosa::TransposeOp>(loc, wOKWICTy, W, pW).getResult();
  
      // ---- 4) Bias: synthesize if missing ----
      bool hasBias = B && !isa<NoneType>(B.getType());
      if (!hasBias) {
        auto biasTy = RankedTensorType::get({OC}, rewriter.getF32Type());
        DenseElementsAttr zeros = DenseElementsAttr::get(biasTy, {0.0f});
        B = rewriter.create<mlir::tosa::ConstOp>(loc, biasTy, zeros);
      }
  
      // ---- 5) Accumulator type ----
      TypeAttr accType = TypeAttr::get(elemTy.isF16() ? rewriter.getF16Type()
                                                      : rewriter.getF32Type());
  
      // ---- 6) Conv result type: derive NHWC from expected NCHW; no shape math needed ----
      auto outNCHW = expectedNCHW.getShape();               // [N,OC,OH,OW]
      SmallVector<int64_t,4> outNHWC{outNCHW[0], outNCHW[2], outNCHW[3], outNCHW[1]};
      auto convOutNHWC = RankedTensorType::get(outNHWC, elemTy);
  
      Value yNHWC;
      Value yNCHW;
      auto pH2NCW = DenseI32ArrayAttr::get(op.getContext(), {0,3,1,2});
      
      if (!quantized){
        yNHWC = rewriter.create<mlir::tosa::Conv2DOp>(loc, convOutNHWC, xNHWC, wOKWICv, B,padsTosa, stridesTosa, dilationsTosa, accType).getResult();
        yNCHW = rewriter.create<mlir::tosa::TransposeOp>(loc, expectedNCHW, yNHWC, pH2NCW).getResult();
      } else {
        
        accType = TypeAttr::get(rewriter.getI32Type());
        yNHWC = rewriter.create<mlir::tosa::Conv2DOp>(loc,RankedTensorType::get(outNHWC, rewriter.getI32Type()), xNHWC, wOKWICv, B, inputZeropoint, weightZeropoint, padsTosa, stridesTosa, dilationsTosa, accType).getResult();
        expectedNCHW = RankedTensorType::get(outNCHW,rewriter.getI32Type());
        yNCHW = rewriter.create<mlir::tosa::TransposeOp>(loc, expectedNCHW, yNHWC, pH2NCW).getResult();
        expectedNCHW = RankedTensorType::get(outNCHW,rewriter.getI8Type());

        bool scale32 = false;
        bool perChannel = false;
        bool inputUnsigned = false;
        bool outputUnsigned = false;
        StringRef s1("SINGLE_ROUND");

        RankedTensorType ri32type = RankedTensorType::get({1},rewriter.getI32Type());
        RankedTensorType ri8type = RankedTensorType::get({1},rewriter.getI8Type());
        DenseElementsAttr inzp = DenseElementsAttr::get(ri32type,ArrayRef<int32_t>({0}));
        DenseElementsAttr ouzp = DenseElementsAttr::get(ri8type, ArrayRef<int8_t>({0}));
        Value reinzp = rewriter.create<mlir::tosa::ConstOp>(loc,ri32type,inzp).getResult();
        Value reouzp = rewriter.create<mlir::tosa::ConstOp>(loc,ri8type,ouzp).getResult();
        Value rescaled = rewriter.create<mlir::tosa::RescaleOp>(loc,expectedNCHW,yNCHW,multVal,shiftVal,reinzp,reouzp, scale32,s1,perChannel,inputUnsigned,outputUnsigned).getResult();
        rewriter.replaceOp(qO,rescaled);
        rewriter.eraseOp(op);
        rewriter.eraseOp(dqX);
        if (dqW.getResult().use_empty())
          rewriter.eraseOp(dqW);
        return success();
      }
  
      
      // ---- 8) Replace ----
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
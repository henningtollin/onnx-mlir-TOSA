


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

class ONNXSoftmaxOpLoweringToTosa : public OpConversionPattern<ONNXSoftmaxOp> { public:
    using OpConversionPattern<ONNXSoftmaxOp>::OpConversionPattern;
    using OpAdaptor = typename ONNXSoftmaxOp::Adaptor;
    LogicalResult matchAndRewrite(ONNXSoftmaxOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {

        Location loc = op.getLoc();
        // 1. We need input j to choose the correct table
        // 2. Make a slice of the table to get row j
        // 3. Compute exp with the slice of the table
        // 4. Do the rest of the softmax
    

        return success();
    }
};


} //namspace
void populateLoweringONNXSoftmaxOpToTOSAPattern(ConversionTarget &target, RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *ctx) {
    patterns.insert<ONNXSoftmaxOpLoweringToTosa>(typeConverter, ctx);
}
} //namespace onnx_mlir
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <tc/core/cpu/cpu_tc_executor.h>
#include "tc/core/polyhedral/scop.h"
#include "tc/core/polyhedral/codegen_llvm.h"
#include "tc/core/polyhedral/cpu/mapped_scop.h"
#include "tc/core/polyhedral/llvm_jit.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut")
    .Input("to_zero: float")
    .Output("zeroed: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    using namespace tc;
    using namespace tc::polyhedral;

    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<float>();

    // Call to TC
  string tc = R"TC(
def fun(float(N) A) -> (C) {
    C(n) = n == 0 ? A(n) : 0.0
}
)TC";
    auto ctx = isl::with_exceptions::globalIslCtx();
    auto scop = polyhedral::Scop::makeScop(ctx, tc);
    const int N = input.size();
    scop = Scop::makeSpecializedScop<int>(*scop, {{"N", N}});
    auto mscop = cpu::MappedScop::makeSequential(
        std::move(scop), CpuMappingOptions::makeNaiveMappingOptions());
    auto pJit = mscop->codegen("kernel_anon");
    auto fptr = reinterpret_cast<void (*)(float*, float*, int)>(
        pJit->getSymbolAddress("kernel_anon"));

    fptr((float *)input.data(), (float *)output_flat.data(), N);
  }
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);


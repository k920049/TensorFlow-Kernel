#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>

using namespace tensorflow;

REGISTER_OP("ZeroOut")
.Attr("T : {float, double, int32}")
.Input("to_zero : T")
.Output("zeroed : T");

template <typename T>
class ZeroOutOp : public OpKernel {
private:

public:

explicit ZeroOutOp(OpKernelConstruction *context) : OpKernel(context) {

}

void Compute(OpKernelContext *context) override {
        const Tensor &input_tensor = context->input(0);
        auto input = input_tensor.flat<T>();

        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output));
        auto output_flat = output->template flat<T>();

        const int N = input.size();

        for (size_t i = 0; i < N; i++) {
                output_flat(i) = 0;
        }

        if (N > 0) {
                output_flat(0) = input(0);
        }
}
};

REGISTER_KERNEL_BUILDER(
  Name("ZeroOut")
  .Device(DEVICE_CPU)
  .TypeConstraint<int32>("T"),
  ZeroOutOp<int32>
);
REGISTER_KERNEL_BUILDER(
  Name("ZeroOut")
  .Device(DEVICE_CPU)
  .TypeConstraint<float>("T"),
  ZeroOutOp<float>
);
REGISTER_KERNEL_BUILDER(
  Name("ZeroOut")
  .Device(DEVICE_CPU)
  .TypeConstraint<double>("T"),
  ZeroOutOp<double>
);

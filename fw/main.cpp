#include "mbed.h"
#include "utensor-model/trained.hpp"
#include "utensor-model/trained_weight.hpp"            // keep the weights in ROM for now, we have plenty of internal flash

#define INFERENCING_OUT_TENSOR_NAME             "y_pred/Softmax:0"

// same as in raw_data.txt
const float raw_data[33] = {
    9.517958, 1.055818, 2.833398, 0.679348, 10.632154, 1.698370, 2.203253, 2.717391,
    0.968113, 1.019022, 0.754527, 2.377717, 0.617696, 3.396739, 0.280435, 0.679348,
    3.185706, 1.358696, 0.884940, 0.000000, 0.000000, 68.825996, 168.506165, 8.870461,
    0.390614, 0.104528, 0.625409, 0.406858, 0.146623, 5.749864, 15.128113, 0.598300,
    0.068765
};

const float empty_data[33] = { 0 };

void classify(const float *data) {
    RamTensor<float> *input_x = new RamTensor<float>({ 1, 33 });
    float *buff = (float*)input_x->write(0, 0);

    for (size_t ix = 0; ix < 33; ix++) {
        buff[ix] = data[ix];
    }

    Context ctx;
    get_trained_ctx(ctx, input_x);
    ctx.eval();

    S_TENSOR pred_tensor = ctx.get(INFERENCING_OUT_TENSOR_NAME);  // getting a reference to the output tensor

    uint32_t output_neurons = pred_tensor->getShape()[1];
    printf("output_neurons %lu\n", output_neurons);

    const float* ptr_pred = pred_tensor->read<float>(0, 0);

    for (uint32_t ix = 0; ix < output_neurons; ix++) {
        if (debug) {
            printf("%lu: %f\n", ix, *(ptr_pred + ix));
        }
    }
}

int main() {
    printf("Raw data:\n");
    classify(raw_data);
    printf("Empty data:\n");
    classify(empty_data);
}

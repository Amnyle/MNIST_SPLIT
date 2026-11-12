#ifndef CONV1_FORWARD_DNN_H
#define CONV1_FORWARD_DNN_H



struct conv1_forward_param {
    float Convolution28_input_0[784];
    float weight[8][1][5][5];
};

struct conv1_forward_res {
    float Convolution28_output_0[6272];
};

void conv1_forward (
    struct conv1_forward_param* param,
    struct conv1_forward_res* result
);


#endif /* CONV1_FORWARD_DNN_H */

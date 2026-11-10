#ifndef FORWARD_DNN_H
#define FORWARD_DNN_H



struct forward_param {
    float Plus30_input_0[6272];
};

struct forward_res {
    float Plus214_output_0[10];
};

void forward (
    struct forward_param* param,
    struct forward_res* result
);


#endif /* FORWARD_DNN_H */

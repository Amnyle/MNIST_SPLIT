#ifndef CONV1_FORWARD_DNN_H
#define CONV1_FORWARD_DNN_H

#include <stdint.h>

/* split of the 8 output channels over the accelerators:
   CONV1_TILE_OC channels per accelerator, CONV1_N_TILES accelerators (8 = 1 x 8) */
#define CONV1_TILE_OC  1
#define CONV1_N_TILES  8

/* whole layer (input image / merged output), used on the CPU side */
struct conv1_forward_param
{
    float Convolution28_Quantizer_0_input_0[784];
};

struct conv1_forward_res
{
    float Convolution28_DeQuantizer_output_0[6272];
};

/* entry of ONE accelerator: the full input image + which tile it computes */
struct conv1_forward_tile_param
{
    float Convolution28_Quantizer_0_input_0[784];   /* same for all the accelerators */
    int32_t tile;                                   /* tile index: 0 .. CONV1_N_TILES-1 */
};

/* result of ONE accelerator: its CONV1_TILE_OC output channels (28 x 28 each) */
struct conv1_forward_tile_res
{
    float Convolution28_DeQuantizer_output_0[CONV1_TILE_OC * 784];
};

/* FPGA function (runs on an accelerator) */
void conv1_forward_tile(struct conv1_forward_tile_param* param, struct conv1_forward_tile_res* result);

#endif /* CONV1_FORWARD_DNN_H */

#include <stdint.h>
#include <string.h>


#include "../include/conv1_forward.h"

#include "stddef.h"

// entry and output structure

// Memory block

void conv1_forward (
    struct conv1_forward_param* param,
    struct conv1_forward_res* result
)
{
    const float* Convolution28_input_0 = param->Convolution28_input_0;
    const float (*weight)[1][5][5] = param->weight;
    
    // Convolution28
    float (*Convolution28_output_0_v)[1][8][28][28] = (float(*)[1][8][28][28]) (result->Convolution28_output_0);
    {
        float (*Convolution28_input_0_v)[1][1][28][28] = (float(*)[1][1][28][28]) (Convolution28_input_0);
        const float (*Parameter5_out0_output_0_v)[8][1][5][5] =
            (const float (*)[8][1][5][5]) weight;
        

        // Conv2D (inputs: 6, outputs 1) with
        // - Input Convolution28_input_0_v[1,1,28,28]
        // - Input (weights) Parameter5_out0_output_0_v[8,1,5,5]
        
        // - Output Convolution28_output_0_v[1,8,28,28]
        // - Stride [1, 1]
        // - Dilation [1, 1]
        for (size_t b = 0; b < 1; ++b) {
            for (size_t c = 0; c < 8; ++c) {
                for (size_t m = 0; m < 28; ++m) {
                    for (size_t n = 0; n < 28; ++n) {
                        
                        float sum = 0;
                        
                        for (size_t i = 0; i < 1; ++i) {
                            for (size_t h = 0; h < 5; ++h) {
                                for (size_t w = 0; w < 5; ++w) {
                                    size_t ii = m*1 + h*1; 
                                    size_t jj = n*1 + w*1; 
                                    if (ii < 2 || jj < 2) {
                                        // Check border on top/left of input
                                        const float input = 0;
                                        const float weight = (*Parameter5_out0_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    } else if (ii - 2 < 28 && jj - 2 < 28) {
                                        const float input = (*Convolution28_input_0_v)[b][i][ii - 2][jj - 2];
                                        const float weight = (*Parameter5_out0_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    } else if ((ii < 2 + 28 + 2) && (jj < 2 + 28 + 2)) {
                                        // Check border on bottom/right of input
                                        const float input = 0;
                                        const float weight = (*Parameter5_out0_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    }
                                }
                            }
                        }
                        (*Convolution28_output_0_v)[b][c][m][n] = sum;
                    }
                }
            }
        }
    }


}
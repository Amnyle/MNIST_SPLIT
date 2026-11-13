#include <stdint.h>
#include <string.h>

#include "../include/conv1_forward.h"

// Layer & memory configurations

#include "../include/parameters/Convolution28_Quantizer_1.h"
#include "stddef.h"

// entry and output structure

void conv1_forward (
    struct conv1_forward_param* param,
    struct conv1_forward_res* result
)
{
    // Local arrays (BRAM), one per buffer, each with its own natural type.
    // This replaces the generic byte pool "static unsigned char mem[203840]".
    int8_t  Convolution28_Quantizer_0_output_0[784];   // was (int8_t*)  (mem + 0)
    int32_t Convolution28_output_0[6272];              // was (int32_t*) (mem + 3136)

    const float* Convolution28_Quantizer_0_input_0 = param->Convolution28_Quantizer_0_input_0;
    
    // Convolution28_Quantizer_0
    {
        for (size_t i = 0; i < 784; ++i) {
            float value = Convolution28_Quantizer_0_input_0[i] * 1.0 + -128.0;
            // rounding written inline (was the helper function locod_roundf)
            value = (float)(value >= 0.0f
                ? (int32_t)(value + 0.5f)
                : (int32_t)(value - 0.5f));
            value = value < -3.4028234663852886e+38 ? -3.4028234663852886e+38 : value;
            value = value > 3.4028234663852886e+38 ? 3.4028234663852886e+38 : value;
            Convolution28_Quantizer_0_output_0[i] = (int8_t)value;
        }
    }


    
    // Convolution28
    {
        

        int32_t (*Convolution28_output_0_v)[1][8][28][28] = (int32_t(*)[1][8][28][28]) (Convolution28_output_0);
        int8_t (*Convolution28_Quantizer_0_output_0_v)[1][1][28][28] = (int8_t(*)[1][1][28][28]) (Convolution28_Quantizer_0_output_0);
        int8_t (*Convolution28_Quantizer_1_output_0_v)[8][1][5][5] = (int8_t(*)[8][1][5][5]) (Convolution28_Quantizer_1_output_0);
        

        // Conv2D (inputs: 6, outputs 1) with
        // - Input Convolution28_Quantizer_0_output_0_v[1,1,28,28]
        // - Input (weights) Convolution28_Quantizer_1_output_0_v[8,1,5,5]
        
        // - Output Convolution28_output_0_v[1,8,28,28]
        // - Stride [1, 1]
        // - Dilation [1, 1]
        for (size_t b = 0; b < 1; ++b) {
            for (size_t c = 0; c < 8; ++c) {
                for (size_t m = 0; m < 28; ++m) {
                    for (size_t n = 0; n < 28; ++n) {
                        
                        int32_t sum = 0;
                        
                        for (size_t i = 0; i < 1; ++i) {
                            for (size_t h = 0; h < 5; ++h) {
                                for (size_t w = 0; w < 5; ++w) {
                                    size_t ii = m*1 + h*1; 
                                    size_t jj = n*1 + w*1; 
                                    if (ii < 2 || jj < 2) {
                                        // Check border on top/left of input
                                        const int32_t input = 0;
                                        const int32_t weight = (*Convolution28_Quantizer_1_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    } else if (ii - 2 < 28 && jj - 2 < 28) {
                                        const int32_t input = (*Convolution28_Quantizer_0_output_0_v)[b][i][ii - 2][jj - 2];
                                        const int32_t weight = (*Convolution28_Quantizer_1_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    } else if ((ii < 2 + 28 + 2) && (jj < 2 + 28 + 2)) {
                                        // Check border on bottom/right of input
                                        const int32_t input = 0;
                                        const int32_t weight = (*Convolution28_Quantizer_1_output_0_v)[c][i][h][w];
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


    
    // Convolution28_DeQuantizer
    // writes straight into result (no float buffer + memcpy anymore)
    {
        for (size_t i = 0; i < 6272; ++i) {
            result->Convolution28_DeQuantizer_output_0[i] = (float)(Convolution28_output_0[i] * 0.008023343048989773 - 0.0);
        }
    }
}
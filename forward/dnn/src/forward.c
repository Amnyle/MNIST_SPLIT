
#include <stdint.h>
#include <string.h>



#include "../include/forward.h"

// Layer & memory configurations


#include "float.h"
#include "../include/parameters/Parameter193_out0.h"
#include "../include/parameters/Parameter193_reshape1_shape_out0.h"
#include "../include/parameters/Parameter194_out0.h"
#include "../include/parameters/Pooling160_Output_0_reshape0_shape_out0.h"
#include "../include/parameters/Parameter6_out0.h"
#include "../include/parameters/Parameter88_out0.h"
#include "../include/parameters/Parameter87_out0.h"
#include "stdint.h"
#include "math.h"
#include "stddef.h"

// entry and output structure

// Memory block
static unsigned char mem[256256];

void forward (
    struct forward_param* param,
    struct forward_res* result
)
{
    const float* Plus30_input_0 = param->Plus30_input_0;
    

    // Times212_reshape1

    //  Reshape tensor from [16, 4, 4, 10] to  ([256, 10])
    float* Times212_reshape1_output_0 = (float*) (mem + 0);
    {
        float (*Times212_reshape1_output_0_v)[2560] = (float(*)[2560]) (Times212_reshape1_output_0);

        float (*Parameter193_out0_output_0_v)[2560] = (float(*)[2560]) (Parameter193_out0_output_0);

        for (size_t i = 0; i < 2560; ++i) {
            (*Times212_reshape1_output_0_v)[i] = (*Parameter193_out0_output_0_v)[i];
        }
    }


    

    // Plus30

    //  Element-wise addition between two input tensors, with broadcasting.
    float* Plus30_output_0 = (float*) (mem + 40960);
    {
        // Element-wise add with
        // - Input Plus30_input_0[1,8,28,28] (broadcast from [1,8,28,28]) (var: Plus30_input_0[6272])
        // - Input Parameter6_out0_output_0[8,1,1] (broadcast from [8,1,1]) (var: Parameter6_out0_output_0[8])
        // - Output Plus30_output_0[1,8,28,28] (var: Plus30_output_0[6272])
        for (size_t i = 0; i < 6272; ++i) {
            // Compute corresponding index in input 0 and input 1

            size_t ia = 0;
            
            ia = i;
        

            size_t ib = 0;
            
        {
            ib = 0;
                size_t r = i;
                size_t a = 1;
            
                // Compute input indices offset in dimension 3 (28 elements in output, 1 in input)
                
                    ib += ((r % 28) % 1) * a;
                
                    // Accumulate offset over traversed dimensions
                    a *= 1;
                    // Compute index over remaining dimensions
                    r = r / 28;
                
                // Compute input indices offset in dimension 2 (28 elements in output, 1 in input)
                
                    ib += ((r % 28) % 1) * a;
                
                    // Accumulate offset over traversed dimensions
                    a *= 1;
                    // Compute index over remaining dimensions
                    r = r / 28;
                
                // Compute input indices offset in dimension 1 (8 elements in output, 8 in input)
                
                    ib += ((r % 8) % 8) * a;
                
                    // Accumulate offset over traversed dimensions
                    a *= 8;
                    // Compute index over remaining dimensions
                    r = r / 8;
                
                // Compute input indices offset in dimension 0 (1 elements in output, 1 in input)
                
        }
        


            Plus30_output_0[i] = Plus30_input_0[ia] + Parameter6_out0_output_0[ib];
        }
    }


    
    // ReLU32

    //  Element-wise RELU activation
    float* ReLU32_output_0 = (float*) (mem + 141312);
    {
        for (unsigned int i = 0; i < 6272; ++i)
        {
            ReLU32_output_0[i] = (Plus30_output_0[i] > 0) ? Plus30_output_0[i] : 0;
        }
    }


    

    /* Pooling66 - MaxPooling2D */

    float* Pooling66_output_0 = (float*) ((uint8_t*)mem + 40960);

    int64_t* Pooling66_output_1 = (int64_t*) ((uint8_t*)mem + 241664);

    {
        const uint32_t N = 1;
        const uint32_t C = 8;
        const uint32_t H = 28;
        const uint32_t W = 28;

        const uint32_t oH = 14;
        const uint32_t oW = 14;

        const int stride_h = 2;
        const int stride_w = 2;

        const int pad_top = 0;
        const int pad_left = 0;

        const int dilation_h = 1;
        const int dilation_w = 1;

        const int kH = 2;
        const int kW = 2;

        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t c = 0; c < C; ++c) {
                for (uint32_t oh = 0; oh < oH; ++oh) {
                    for (uint32_t ow = 0; ow < oW; ++ow) {

                        float max_val = -FLT_MAX;
                        int64_t max_idx = 0;
                        int found = 0;
                        // Origine de la fenêtre AVANT padding
                        int base_h = oh * stride_h - pad_top;
                        int base_w = ow * stride_w - pad_left;

                        //for (int kh = 0; kh < kH; ++kh) { // ONNX RUNTIME
                        //    for (int kw = 0; kw < kW; ++kw) { // ONNX RUNTIME
    		      for (int kw = 0; kw < kW; ++kw) { //AIDGE CPU
    		        for (int kh = 0; kh < kH; ++kh) { //AIDGE CPU


                                int ih = base_h + kh * dilation_h;
                                int iw = base_w + kw * dilation_w;

                                // IGNORER le padding (ONNX compliant)
                                if (ih < 0 || ih >= (int)H || iw < 0 || iw >= (int)W) {
                                    continue;
                                }
                                uint32_t in_idx = (n * C * H * W) + (c * H * W) + (ih * W) + (uint32_t)iw;

                                float val = (float)ReLU32_output_0[in_idx];

                                if (!found || val > max_val) {
                                    max_val = val;
                                    // INDICE GLOBAL (flatten NCHW ONNX)
                                    max_idx = (int64_t)in_idx;
                                    found = 1;
                                }
                            }
                        }

                        // cas extrême : fenetre entierement dans le padding
                        if (!found) {
                            max_val = -FLT_MAX;
                            max_idx = 0;

                        }

                        uint32_t out_idx = (n * C * oH * oW) + (c * oH * oW) + (oh * oW) + ow;

                        Pooling66_output_0[out_idx] = (float)max_val;
                        
                        Pooling66_output_1[out_idx] = (int64_t)max_idx;
                        
                    }
                }
            }
        }
    }




    
    // Convolution110
    float *Convolution110_output_0 = (float*) (mem + 66048);
    {
        

        float (*Convolution110_output_0_v)[1][16][14][14] = (float(*)[1][16][14][14]) (mem + 66048);
        float (*Pooling66_output_0_v)[1][8][14][14] = (float(*)[1][8][14][14]) (Pooling66_output_0);
        float (*Parameter87_out0_output_0_v)[16][8][5][5] = (float(*)[16][8][5][5]) (Parameter87_out0_output_0);
        

        // Conv2D (inputs: 6, outputs 1) with
        // - Input Pooling66_output_0_v[1,8,14,14]
        // - Input (weights) Parameter87_out0_output_0_v[16,8,5,5]
        
        // - Output Convolution110_output_0_v[1,16,14,14]
        // - Stride [1, 1]
        // - Dilation [1, 1]
        for (size_t b = 0; b < 1; ++b) {
            for (size_t c = 0; c < 16; ++c) {
                for (size_t m = 0; m < 14; ++m) {
                    for (size_t n = 0; n < 14; ++n) {
                        
                        float sum = 0;
                        
                        for (size_t i = 0; i < 8; ++i) {
                            for (size_t h = 0; h < 5; ++h) {
                                for (size_t w = 0; w < 5; ++w) {
                                    size_t ii = m*1 + h*1; 
                                    size_t jj = n*1 + w*1; 
                                    if (ii < 2 || jj < 2) {
                                        // Check border on top/left of input
                                        const float input = 0;
                                        const float weight = (*Parameter87_out0_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    } else if (ii - 2 < 14 && jj - 2 < 14) {
                                        const float input = (*Pooling66_output_0_v)[b][i][ii - 2][jj - 2];
                                        const float weight = (*Parameter87_out0_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    } else if ((ii < 2 + 14 + 2) && (jj < 2 + 14 + 2)) {
                                        // Check border on bottom/right of input
                                        const float input = 0;
                                        const float weight = (*Parameter87_out0_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    }
                                }
                            }
                        }
                        (*Convolution110_output_0_v)[b][c][m][n] = sum;
                    }
                }
            }
        }
    }


    

    // Plus112

    //  Element-wise addition between two input tensors, with broadcasting.
    float* Plus112_output_0 = (float*) (mem + 116224);
    {
        // Element-wise add with
        // - Input Convolution110_output_0[1,16,14,14] (broadcast from [1,16,14,14]) (var: Convolution110_output_0[3136])
        // - Input Parameter88_out0_output_0[16,1,1] (broadcast from [16,1,1]) (var: Parameter88_out0_output_0[16])
        // - Output Plus112_output_0[1,16,14,14] (var: Plus112_output_0[3136])
        for (size_t i = 0; i < 3136; ++i) {
            // Compute corresponding index in input 0 and input 1

            size_t ia = 0;
            
            ia = i;
        

            size_t ib = 0;
            
        {
            ib = 0;
                size_t r = i;
                size_t a = 1;
            
                // Compute input indices offset in dimension 3 (14 elements in output, 1 in input)
                
                    ib += ((r % 14) % 1) * a;
                
                    // Accumulate offset over traversed dimensions
                    a *= 1;
                    // Compute index over remaining dimensions
                    r = r / 14;
                
                // Compute input indices offset in dimension 2 (14 elements in output, 1 in input)
                
                    ib += ((r % 14) % 1) * a;
                
                    // Accumulate offset over traversed dimensions
                    a *= 1;
                    // Compute index over remaining dimensions
                    r = r / 14;
                
                // Compute input indices offset in dimension 1 (16 elements in output, 16 in input)
                
                    ib += ((r % 16) % 16) * a;
                
                    // Accumulate offset over traversed dimensions
                    a *= 16;
                    // Compute index over remaining dimensions
                    r = r / 16;
                
                // Compute input indices offset in dimension 0 (1 elements in output, 1 in input)
                
        }
        


            Plus112_output_0[i] = Convolution110_output_0[ia] + Parameter88_out0_output_0[ib];
        }
    }


    
    // ReLU114

    //  Element-wise RELU activation
    float* ReLU114_output_0 = (float*) (mem + 40960);
    {
        for (unsigned int i = 0; i < 3136; ++i)
        {
            ReLU114_output_0[i] = (Plus112_output_0[i] > 0) ? Plus112_output_0[i] : 0;
        }
    }


    

    /* Pooling160 - MaxPooling2D */

    float* Pooling160_output_0 = (float*) ((uint8_t*)mem + 91136);

    int64_t* Pooling160_output_1 = (int64_t*) ((uint8_t*)mem + 254208);

    {
        const uint32_t N = 1;
        const uint32_t C = 16;
        const uint32_t H = 14;
        const uint32_t W = 14;

        const uint32_t oH = 4;
        const uint32_t oW = 4;

        const int stride_h = 3;
        const int stride_w = 3;

        const int pad_top = 0;
        const int pad_left = 0;

        const int dilation_h = 1;
        const int dilation_w = 1;

        const int kH = 3;
        const int kW = 3;

        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t c = 0; c < C; ++c) {
                for (uint32_t oh = 0; oh < oH; ++oh) {
                    for (uint32_t ow = 0; ow < oW; ++ow) {

                        float max_val = -FLT_MAX;
                        int64_t max_idx = 0;
                        int found = 0;
                        // Origine de la fenêtre AVANT padding
                        int base_h = oh * stride_h - pad_top;
                        int base_w = ow * stride_w - pad_left;

                        //for (int kh = 0; kh < kH; ++kh) { // ONNX RUNTIME
                        //    for (int kw = 0; kw < kW; ++kw) { // ONNX RUNTIME
    		      for (int kw = 0; kw < kW; ++kw) { //AIDGE CPU
    		        for (int kh = 0; kh < kH; ++kh) { //AIDGE CPU


                                int ih = base_h + kh * dilation_h;
                                int iw = base_w + kw * dilation_w;

                                // IGNORER le padding (ONNX compliant)
                                if (ih < 0 || ih >= (int)H || iw < 0 || iw >= (int)W) {
                                    continue;
                                }
                                uint32_t in_idx = (n * C * H * W) + (c * H * W) + (ih * W) + (uint32_t)iw;

                                float val = (float)ReLU114_output_0[in_idx];

                                if (!found || val > max_val) {
                                    max_val = val;
                                    // INDICE GLOBAL (flatten NCHW ONNX)
                                    max_idx = (int64_t)in_idx;
                                    found = 1;
                                }
                            }
                        }

                        // cas extrême : fenetre entierement dans le padding
                        if (!found) {
                            max_val = -FLT_MAX;
                            max_idx = 0;

                        }

                        uint32_t out_idx = (n * C * oH * oW) + (c * oH * oW) + (oh * oW) + ow;

                        Pooling160_output_0[out_idx] = (float)max_val;
                        
                        Pooling160_output_1[out_idx] = (int64_t)max_idx;
                        
                    }
                }
            }
        }
    }




    

    // Times212_reshape0

    //  Reshape tensor from [1, 16, 4, 4] to  ([1, 256])
    float* Times212_reshape0_output_0 = (float*) (mem + 40960);
    {
        float (*Times212_reshape0_output_0_v)[256] = (float(*)[256]) (Times212_reshape0_output_0);

        float (*Pooling160_output_0_v)[256] = (float(*)[256]) (Pooling160_output_0);

        for (size_t i = 0; i < 256; ++i) {
            (*Times212_reshape0_output_0_v)[i] = (*Pooling160_output_0_v)[i];
        }
    }


    

    // Times212
    float* Times212_output_0 = (float*) (mem + 45056);
    {
        float (*Times212_output_0_v)[1][1][10] = (float(*)[1][1][10]) (mem + 45056);
        float (*Times212_reshape0_output_0_v)[1][1][256] = (float(*)[1][1][256]) (Times212_reshape0_output_0);
        float (*Times212_reshape1_output_0_v)[1][256][10] = (float(*)[1][256][10]) (Times212_reshape1_output_0);
        // Matrix multiplication with
        // - Input Times212_reshape0_output_0_v[1,1,256] (from [1,256])
        // - Input Times212_reshape1_output_0_v[1,256,10] (from [256,10 ])
        // - Output Times212_output_0_v[1,1,10] (from [1,10])

        // Output element accumulator
        float Times212_matmul;
        // Iterate over non-compute dimensions
        for (size_t i = 0; i < 1; ++i)
        {
            size_t ia;
            
        {
            ia = 0;
                size_t r = i;
                size_t a = 1;
            
                // Compute input indices offset in dimension 0 (1 elements in output, 1 in input)
                
        }
        

            size_t ib;
            
        {
            ib = 0;
                size_t r = i;
                size_t a = 1;
            
                // Compute input indices offset in dimension 0 (1 elements in output, 1 in input)
                
        }
        

            // Compute matrix multiplication
            for (size_t m = 0; m < 1; ++m)
            {
                for (size_t n = 0; n < 10; ++n)
                {
                    Times212_matmul = 0;
                    for (size_t p = 0; p < 256; ++p)
                    {
                        Times212_matmul += (*Times212_reshape0_output_0_v)[ia][m][p] * (*Times212_reshape1_output_0_v)[ib][p][n];
                    }
                    (*Times212_output_0_v)[i][m][n] = Times212_matmul;
                }
            }
        }

    }


    

    // Plus214

    //  Element-wise addition between two input tensors, with broadcasting.
    float* Plus214_output_0 = (float*) (mem + 0);
    {
        // Element-wise add with
        // - Input Times212_output_0[1,10] (broadcast from [1,10]) (var: Times212_output_0[10])
        // - Input Parameter194_out0_output_0[1,10] (broadcast from [1,10]) (var: Parameter194_out0_output_0[10])
        // - Output Plus214_output_0[1,10] (var: Plus214_output_0[10])
        for (size_t i = 0; i < 10; ++i) {
            // Compute corresponding index in input 0 and input 1

            size_t ia = 0;
            
            ia = i;
        

            size_t ib = 0;
            
            ib = i;
        


            Plus214_output_0[i] = Times212_output_0[ia] + Parameter194_out0_output_0[ib];
        }
    }


    memcpy(result->Plus214_output_0, Plus214_output_0, sizeof(result->Plus214_output_0));
}

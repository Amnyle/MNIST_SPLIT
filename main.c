#include <stddef.h>
#include <stdio.h>
#include <string.h>

#ifndef LOCOD_FPGA
#include "locod.h"
#include "conv/dnn/include/parameters/Convolution28_Quantizer_1.h"
#endif //LOCOD_FPGA

#include "conv/dnn/include/conv1_forward.h"  
#include "forward/dnn/include/forward.h"      


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


/* input image, taken from the model: 1 x 1 x 28 x 28 (N x C x H x W) */
#define IMG_C 1
#define IMG_H 28
#define IMG_W 28


static int load_image(const char* path, float* data)
{
    int width, height, channels;

    unsigned char* img = stbi_load(path, &width, &height, &channels, IMG_C);

    if (!img) {
        fprintf(stderr, "Failed to load image '%s': %s\n", path, stbi_failure_reason());
        return 1;
    }

    if (width != IMG_W || height != IMG_H) {
        fprintf(stderr, "Expected %dx%d image, got %dx%d\n", IMG_W, IMG_H, width, height);
        stbi_image_free(img);
        return 1;
    }

    for (int y = 0; y < IMG_H; ++y) {
        for (int x = 0; x < IMG_W; ++x) {
            for (int c = 0; c < IMG_C; ++c) {
                /* quantized model: Quantizer_0 expects raw pixel values 0..255 (x * 1.0 - 128), no / 255 */
                data[c * IMG_H * IMG_W + y * IMG_W + x] =
                    (float)img[(y * IMG_W + x) * IMG_C + c];
            }
        }
    }

    stbi_image_free(img);

    return 0;
}

#ifndef LOCOD_FPGA
int main(int argc, char* argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <image>\n", argv[0]);
        return 1;
    }

    static struct conv1_forward_param entry_2;       /* the image */
    static struct conv1_forward_res output_2;        /* merged output of the 8 tiles */
    static struct conv1_forward_tile_param entry_tile[CONV1_N_TILES];
    static struct conv1_forward_tile_res res_tile[CONV1_N_TILES];
    static struct forward_param entry_3;
    static struct forward_res output_3;
  

    if (load_image(argv[1], (float*) entry_2.Convolution28_Quantizer_0_input_0) != 0) {
        return 1;
    }

    /* the number of accelerators must be a literal for locod.sh */
    init_locod(8);

    /* model 2: first conv, split over 8 accelerators (1 output channel each) */
    for (int t = 0; t < CONV1_N_TILES; ++t) {
        memcpy(entry_tile[t].Convolution28_Quantizer_0_input_0,
               entry_2.Convolution28_Quantizer_0_input_0,
               sizeof(entry_tile[t].Convolution28_Quantizer_0_input_0));
        entry_tile[t].tile = t;
    }

    /* one line per accelerator, accelerator number = literal at the end (locod.sh reads it) */
    FPGA(conv1_forward_tile, &entry_tile[0], &res_tile[0], 0);
    FPGA(conv1_forward_tile, &entry_tile[1], &res_tile[1], 1);
    FPGA(conv1_forward_tile, &entry_tile[2], &res_tile[2], 2);
    FPGA(conv1_forward_tile, &entry_tile[3], &res_tile[3], 3);
    FPGA(conv1_forward_tile, &entry_tile[4], &res_tile[4], 4);
    FPGA(conv1_forward_tile, &entry_tile[5], &res_tile[5], 5);
    FPGA(conv1_forward_tile, &entry_tile[6], &res_tile[6], 6);
    FPGA(conv1_forward_tile, &entry_tile[7], &res_tile[7], 7);

    wait_accelerator(&res_tile[0], 0);
    wait_accelerator(&res_tile[1], 1);
    wait_accelerator(&res_tile[2], 2);
    wait_accelerator(&res_tile[3], 3);
    wait_accelerator(&res_tile[4], 4);
    wait_accelerator(&res_tile[5], 5);
    wait_accelerator(&res_tile[6], 6);
    wait_accelerator(&res_tile[7], 7);

    /* merge: tile t = output channels t*CONV1_TILE_OC ... (each channel is 28 x 28 floats) */
    for (int t = 0; t < CONV1_N_TILES; ++t) {
        memcpy(output_2.Convolution28_DeQuantizer_output_0 + t * CONV1_TILE_OC * 784,
               res_tile[t].Convolution28_DeQuantizer_output_0,
               sizeof(res_tile[t].Convolution28_DeQuantizer_output_0));
    }

    
    /* model 2 output -> model 3 input (both 1 x 8 x 28 x 28 floats) */
    memcpy(entry_3.Plus30_input_0, output_2.Convolution28_DeQuantizer_output_0, sizeof(entry_3.Plus30_input_0));

    /* model 3: everything after the first conv */
    CPU(forward, &entry_3, &output_3);

    deinit_locod();

    for (size_t i = 0; i < 10; ++i) {
        printf("%.10f ", output_3.Plus214_output_0[i]);
    }
    printf("\n");

    return 0;
}
#endif //LOCOD_FPGA
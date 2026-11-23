# MNIST-12 execution log

Median time per image, over the same 1088 images in every run: 12.669 ms on CPU only (run 1), 15.100 ms with the quantized conv split across 8 FPGA accelerators in the latest version (run 8).

## MNIST-12 run log

| Run | Date | Setup | Commit | Median (ms) | Median vs run 1 | Mean (ms) | Min (ms) | Max (ms) | Total (ms) | Accuracy (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 2026-11-13 | Same as run 7, with the conv split across 8 accelerators | RUN 8 | 15.100 | 1.19× | 20.590 | 14.897 | 1004.416 | 22401.8 | 97.8 |
| 7 | 2026-11-13 | Quantized conv on FPGA, parameters stored in a local array before executing | RUN 7 | 23.992 | 1.89× | 27.453 | 23.817 | 1052.509 | 29868.4 | 97.8 |
| 6 | 2026-11-13 | Quantized conv, run on CPU (no writing in a local array) | RUN 6 | 12.831 | 1.01× | 15.132 | 12.668 | 478.581 | 16463.9 | 97.8 |
| 5 | 2026-11-12 | Same as run 4, after the LoCod fix; weights no longer passed as parameters | RUN 5 | 39.337 | 3.10× | 43.287 | 39.192 | 2061.111 | 47095.9 | 99.3 |
| 4 | 2026-11-12 | Same as run 3, with the arrays stored locally before executing | COMMIT 4 | 38.924 | 3.07× | 42.822 | 38.805 | 1549.375 | 46590.4 | 99.3 |
| 3 | 2026-11-10 | Same code as run 2; first conv launched on FPGA (rest on CPU); conv header added as parameters to the function | RUN 3 | 148.947 | 11.76× | 154.724 | 148.825 | 625.052 | 168339.3 | 99.3 |
| 2 | 2026-11-09 | CPU only, but model_forward is split in two functions |  | 12.914 | 1.02× | 14.319 | 12.735 | 651.463 | 15579.5 | 99.3 |
| 1 | 2026-11-09 | CPU only, no optimisation |  | 12.669 | 1.00× | 12.789 | 12.520 | 68.470 | 13914.2 | 99.3 |

## MNIST-12 common setup

| Parameter | Value |
| --- | --- |
| Model | MNIST-12 |
| Images per run | 1088 |
| Board | Enclustra |

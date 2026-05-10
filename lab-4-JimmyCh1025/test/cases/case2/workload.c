/*
 * Workload: qconv2d + ReLU + MaxPool(2x2)
 *   Input : N=1 C=64  H=16 W=16
 *   Kernel: K=192 C=64 H=3  W=3
 *   Output: N=1 K=192 H=8  W=8   (12288 bytes, after maxpool)
 */

#include "workload.h"

#include "data.h"
#include "runtime.h"

uint8_t output[12288];  // 1 * 192 * 8 * 8

void run_workload(void) {
#ifdef DLA_BACKEND
    qconv2d_relu_maxpool(input_tensor, weight_tensor, output, bias_tensor,
                         (192 * 8 * 8), (64 * 16 * 16), (192 * 64 * 3 * 3), 64,
                         DEFAULT_e, DEFAULT_p, DEFAULT_q, DEFAULT_r, DEFAULT_t,
                         1, 1, 3, 3, 64, 192, 16, 16, 7);
#else
    qconv2d_relu_maxpool_cpu(input_tensor, weight_tensor, output, bias_tensor,
                             (192 * 8 * 8), (64 * 16 * 16), (192 * 64 * 3 * 3),
                             1, 1, 3, 3, 64, 192, 16, 16, 7);
#endif
}

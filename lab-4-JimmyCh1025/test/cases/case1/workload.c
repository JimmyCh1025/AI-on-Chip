/*
 * Workload: qconv2d + ReLU + MaxPool(2x2)
 *   Input : N=1 C=3  H=32 W=32
 *   Kernel: K=64 C=3 H=3  W=3
 *   Output: N=1 K=64 H=16 W=16  (16384 bytes, after maxpool)
 */

#include "workload.h"

#include "data.h"
#include "runtime.h"

uint8_t output[16384];  // 1 * 64 * 16 * 16

void run_workload(void) {
#ifdef DLA_BACKEND
    qconv2d_relu_maxpool(activation_flat_array, nchw_flat_array, output, bias,
                         (64 * 16 * 16), (3 * 32 * 32), (64 * 3 * 3 * 3), 64,
                         DEFAULT_e, DEFAULT_p, DEFAULT_q, DEFAULT_r, DEFAULT_t,
                         1, 1, 3, 3, 3, 64, 32, 32, 8);
#else
    qconv2d_relu_maxpool_cpu(activation_flat_array, nchw_flat_array, output,
                             bias, (64 * 16 * 16), (3 * 32 * 32),
                             (64 * 3 * 3 * 3), 1, 1, 3, 3, 3, 64, 32, 32, 8);
#endif
}

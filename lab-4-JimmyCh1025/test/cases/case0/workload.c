/*
 * Workload: qconv2d + ReLU
 *   Input : N=1 C=3  H=8  W=8
 *   Kernel: K=16 C=3 H=3  W=3
 *   Output: N=1 K=16 H=8  W=8  (1024 bytes)
 */

#include "workload.h"

#include "data.h"
#include "runtime.h"

uint8_t output[1024];  // 1 * 16 * 8 * 8

void run_workload(void) {
#ifdef DLA_BACKEND
    qconv2d_relu(activation_flat_array, nchw_flat_array, output, bias,
                 (8 * 8 * 16), (8 * 8 * 3), (16 * 3 * 3 * 3), 64, DEFAULT_e,
                 DEFAULT_p, DEFAULT_q, DEFAULT_r, DEFAULT_t, 1, 1, 3, 3, 3, 16,
                 8, 8, 8);
#else
    qconv2d_relu_cpu(activation_flat_array, nchw_flat_array, output, bias,
                     (8 * 8 * 16), (8 * 8 * 3), (16 * 3 * 3 * 3), 1, 1, 3, 3, 3,
                     16, 8, 8, 8);
#endif
}

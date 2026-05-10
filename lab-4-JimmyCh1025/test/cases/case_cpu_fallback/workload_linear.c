/*
 * Workload: quantized linear (no activation)
 *   Input : 4096 elements (uint8)
 *   Weight: 4096×256 = 1048576 elements (int8)
 *   Output: 256 elements (uint8)
 */

#include "data.h"
#include "runtime.h"
#include "workload.h"

uint8_t output[256];  // 256

void run_workload(void) {
    qlinear_cpu(linear_activation, linear_weight, output, linear_bias,
                linear_out_len, linear_in_len, linear_weight_len,
                linear_q_scale);
}

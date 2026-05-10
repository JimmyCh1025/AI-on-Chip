#include "kernel_cpu.h"
#include "runtime.h"

#if CASE_ID == 0
#define SCRATCH_SIZE 300
#elif CASE_ID == 1
#define SCRATCH_SIZE 3468
#elif CASE_ID == 2 
#define SCRATCH_SIZE 20736
#elif CASE_ID == 3
#define SCRATCH_SIZE (4096 * 4)
#elif CASE_ID == 4
#define SCRATCH_SIZE (4096 * 4)
#endif

static uint8_t _scratch[SCRATCH_SIZE];

/* Repack weights from DLA layout (ceil4(C)*R*S per filter) to compact
 * CPU layout (C*R*S per filter), in-place. Safe since compact ≤ DLA size. */
static void dla_to_compact_weights(int8_t* w, uint32_t M, uint32_t C,
                                   uint32_t R, uint32_t S) {
    uint32_t C4 = (C + 3) & ~3u;
    uint32_t crs = C * R * S;
    uint32_t c4rs = C4 * R * S;
    if (crs == c4rs) return;
    for (uint32_t n = 0; n < M; n++) {
        for (uint32_t c = 0; c < C; c++) {
            for (uint32_t i = 0; i < R * S; i++) {
                w[n * crs + c * R * S + i] = w[n * c4rs + c * R * S + i];
            }
        }
    }
}

void qconv2d_relu_maxpool_cpu(uint8_t* input_in_DRAM, int8_t* filter_in_DRAM,
                              uint8_t* opsum_in_DRAM, int32_t* bias,
                              uint32_t ofmap_len, uint32_t ifmap_len,
                              uint32_t filter_len, uint32_t PAD, uint32_t U,
                              uint32_t R, uint32_t S, uint32_t C, uint32_t M,
                              uint32_t W, uint32_t H, uint32_t scale) {
    dla_to_compact_weights(filter_in_DRAM, M, C, R, S);
    conv_maxpooling(C, H, W, input_in_DRAM, M, C, R, S, filter_in_DRAM, bias,
                    PAD, opsum_in_DRAM, scale, _scratch);
};

void qconv2d_relu_cpu(uint8_t* input_in_DRAM, int8_t* filter_in_DRAM,
                      uint8_t* opsum_in_DRAM, int32_t* bias, uint32_t ofmap_len,
                      uint32_t ifmap_len, uint32_t filter_len, uint32_t PAD,
                      uint32_t U, uint32_t R, uint32_t S, uint32_t C,
                      uint32_t M, uint32_t W, uint32_t H, uint32_t scale) {
    dla_to_compact_weights(filter_in_DRAM, M, C, R, S);
    conv(C, H, W, input_in_DRAM, M, C, R, S, filter_in_DRAM, bias, PAD,
         opsum_in_DRAM, scale, _scratch);
};

void qlinear_relu_cpu(uint8_t* input_in_DRAM, int8_t* filter_in_DRAM,
                      uint8_t* opsum_in_DRAM, int32_t* bias, uint32_t ofmap_len,
                      uint32_t ifmap_len, uint32_t filter_len, uint32_t scale) {
    linear_relu(ifmap_len, ofmap_len, input_in_DRAM, opsum_in_DRAM,
                filter_in_DRAM, bias, scale, _scratch);
};

void qlinear_cpu(uint8_t* input_in_DRAM, int8_t* filter_in_DRAM,
                 uint8_t* opsum_in_DRAM, int32_t* bias, uint32_t ofmap_len,
                 uint32_t ifmap_len, uint32_t filter_len, uint32_t scale) {
    linear(ifmap_len, ofmap_len, input_in_DRAM, opsum_in_DRAM, filter_in_DRAM,
           bias, scale, _scratch);
};

#include "kernel_cpu.h"
#include "runtime.h"

/* Repack weights from DLA layout (ceil4(C)*R*S per filter) to compact
 * CPU layout (C*R*S per filter), in-place. Safe since compact ≤ DLA size. */
static void dla_to_compact_weights(int8_t* w, uint32_t M, uint32_t C,
                                   uint32_t R, uint32_t S) {
    uint32_t C4 = (C + 3) & ~3u; /* ceil4(C) */
    uint32_t crs = C * R * S;    /* compact stride per filter */
    uint32_t c4rs = C4 * R * S;  /* DLA stride per filter */
    if (crs == c4rs) return; /* no padding needed (C already multiple of 4) */
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
                    PAD, opsum_in_DRAM, scale);
};

void qconv2d_relu_cpu(uint8_t* input_in_DRAM, int8_t* filter_in_DRAM,
                      uint8_t* opsum_in_DRAM, int32_t* bias, uint32_t ofmap_len,
                      uint32_t ifmap_len, uint32_t filter_len, uint32_t PAD,
                      uint32_t U, uint32_t R, uint32_t S, uint32_t C,
                      uint32_t M, uint32_t W, uint32_t H, uint32_t scale) {
    dla_to_compact_weights(filter_in_DRAM, M, C, R, S);
    conv(C, H, W, input_in_DRAM, M, C, R, S, filter_in_DRAM, bias, PAD,
         opsum_in_DRAM, scale);
};

void qlinear_relu_cpu(uint8_t* input_in_DRAM, int8_t* filter_in_DRAM,
                      uint8_t* opsum_in_DRAM, int32_t* bias, uint32_t ofmap_len,
                      uint32_t ifmap_len, uint32_t filter_len, uint32_t scale) {
    linear_relu(ifmap_len, ofmap_len, input_in_DRAM, opsum_in_DRAM,
                filter_in_DRAM, bias, scale);
};

void qlinear_cpu(uint8_t* input_in_DRAM, int8_t* filter_in_DRAM,
                 uint8_t* opsum_in_DRAM, int32_t* bias, uint32_t ofmap_len,
                 uint32_t ifmap_len, uint32_t filter_len, uint32_t scale) {
    linear(ifmap_len, ofmap_len, input_in_DRAM, opsum_in_DRAM, filter_in_DRAM,
           bias, scale);
};
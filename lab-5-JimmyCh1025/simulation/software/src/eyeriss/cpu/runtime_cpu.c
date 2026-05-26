#include "hardware_cpu.h"
#include "runtime.h"

int qconv2d_relu_maxpool_cpu(uint8_t* input_in_DRAM, int8_t* filter_in_DRAM, uint8_t* opsum_in_DRAM, int32_t* bias,
                             uint32_t ofmap_len, uint32_t ifmap_len, uint32_t filter_len,
                             // shape parameter
                             uint32_t PAD, uint32_t U, uint32_t R, uint32_t S, uint32_t C, uint32_t M, uint32_t W,
                             uint32_t H,
                             uint32_t scale) {  // int32_t scale_factor: merge ifmap and weight and
                                                // ofmap scale bit-shift
  conv_maxpooling(C, H, W, input_in_DRAM, M, C, R, S, filter_in_DRAM, bias, PAD, opsum_in_DRAM, scale);
  return 0;
};

int qconv2d_relu_cpu(uint8_t* input_in_DRAM, int8_t* filter_in_DRAM, uint8_t* opsum_in_DRAM, int32_t* bias,
                     uint32_t ofmap_len, uint32_t ifmap_len, uint32_t filter_len,
                     // shape parameter
                     uint32_t PAD, uint32_t U, uint32_t R, uint32_t S, uint32_t C, uint32_t M, uint32_t W, uint32_t H,
                     uint32_t scale) {  // int32_t scale_factor: merge ifmap and
                                        // weight and ofmap scale bit-shift
  conv(C, H, W, input_in_DRAM, M, C, R, S, filter_in_DRAM, bias, PAD, opsum_in_DRAM, scale);
  return 0;
};

int qlinear_relu_cpu(uint8_t* input_in_DRAM, int8_t* filter_in_DRAM, uint8_t* opsum_in_DRAM, int32_t* bias,
                     uint32_t ofmap_len, uint32_t ifmap_len, uint32_t filter_len,
                     uint32_t scale) {  // int32_t scale_factor: merge ifmap and
                                        // weight and ofmap scale bit-shift
  linear_relu(ifmap_len, ofmap_len, input_in_DRAM, opsum_in_DRAM, filter_in_DRAM, bias, scale);
  return 0;
};

int qlinear_cpu(uint8_t* input_in_DRAM, int8_t* filter_in_DRAM, uint8_t* opsum_in_DRAM, int32_t* bias,
                uint32_t ofmap_len, uint32_t ifmap_len, uint32_t filter_len,
                uint32_t scale) {  // int32_t scale_factor: merge ifmap and
                                   // weight and ofmap scale bit-shift
  linear(ifmap_len, ofmap_len, input_in_DRAM, opsum_in_DRAM, filter_in_DRAM, bias, scale);
  return 0;
};

void quantize_cpu(float* input_in_DRAM, uint8_t* output_in_DRAM, uint32_t size, uint32_t scale) {
  quantize(input_in_DRAM, output_in_DRAM, size, scale);
};

void dequantize_cpu(uint8_t* input_in_DRAM, float* output_in_DRAM, uint32_t size, uint32_t scale) {
  dequantize(input_in_DRAM, output_in_DRAM, size, scale);
};

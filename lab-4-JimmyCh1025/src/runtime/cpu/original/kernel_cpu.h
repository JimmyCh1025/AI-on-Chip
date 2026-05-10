#ifndef KERNEL_CPU_H
#define KERNEL_CPU_H

#include <stdint.h>
//#include <string.h>

// ReLU: clamp to 0 if negative
#define relu(x) ((int32_t)(x) < 0 ? 0 : (uint32_t)(x))

// Requantize: right-shift by scale, add 128, clamp to [0,255]
#define requant(input, scale)                                \
    (((scale) >= 32) ? 128                                   \
                     : (((((input) >> (scale)) + 128) > 255) \
                            ? 255                            \
                            : ((uint8_t)(((input) >> (scale)) + 128))))

void conv_maxpooling(uint32_t input_C, uint32_t input_H, uint32_t input_W,
                     uint8_t* activation, uint32_t filter_N, uint32_t filter_C,
                     uint32_t filter_H, uint32_t filter_W, int8_t* filter,
                     int32_t* bias, uint32_t padding, uint8_t* output,
                     uint32_t scale);

void conv(uint32_t input_C, uint32_t input_H, uint32_t input_W,
          uint8_t* activation, uint32_t filter_N, uint32_t filter_C,
          uint32_t filter_H, uint32_t filter_W, int8_t* filter, int32_t* bias,
          uint32_t padding, uint8_t* output, uint32_t scale);

void linear_relu(uint32_t input_size, uint32_t output_size, uint8_t* activation,
                 uint8_t* output, int8_t* filter, int32_t* bias,
                 uint32_t scale);

void linear(uint32_t input_size, uint32_t output_size, uint8_t* activation,
            uint8_t* output, int8_t* filter, int32_t* bias, uint32_t scale);

#endif  // KERNEL_CPU_H

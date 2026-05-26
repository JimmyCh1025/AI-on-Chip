#ifndef HARDWARE_CPU_H
#define HARDWARE_CPU_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ReLU activation function: If x < 0, return 0; otherwise, return x
#define relu(x) ((int32_t)(x) < 0 ? 0 : (uint32_t)(x))

// Requantization macro: Adjusts input using a scale factor with clamping
#define requant(input, scale) \
  (((scale) >= 32) ? 128 : (((((input) >> (scale)) + 128) > 255) ? 255 : ((uint8_t)(((input) >> (scale)) + 128))))

/** parameters
 * @param input_C      Number of input channels.
 * @param input_H      Input height.
 * @param input_W      Input width.
 * @param activation   Pointer to input activation data.
 * @param filter_N     Number of output filters.
 * @param filter_C     Number of filter channels.
 * @param filter_H     Filter height.
 * @param filter_W     Filter width.
 * @param filter       Pointer to filter weights.
 * @param bias         Pointer to bias values.
 * @param padding      Padding size.
 * @param output       Pointer to output buffer.
 * @param scale        Scale factor for quantization.
 */

/**
 * @brief Performs convolution followed by max pooling.
 */
void conv_maxpooling(uint32_t input_C, uint32_t input_H, uint32_t input_W, uint8_t* activation, uint32_t filter_N,
                     uint32_t filter_C, uint32_t filter_H, uint32_t filter_W, int8_t* filter, int32_t* bias,
                     uint32_t padding, uint8_t* output, uint32_t scale);

/**
 * @brief Performs standard convolution operation.
 */
void conv(uint32_t input_C, uint32_t input_H, uint32_t input_W, uint8_t* activation, uint32_t filter_N,
          uint32_t filter_C, uint32_t filter_H, uint32_t filter_W, int8_t* filter, int32_t* bias, uint32_t padding,
          uint8_t* output, uint32_t scale);

/**
 * @brief Performs a fully connected (linear) layer with ReLU activation.
 */
void linear_relu(uint32_t input_size, uint32_t output_size, uint8_t* activation, uint8_t* output, int8_t* filter,
                 int32_t* bias, uint32_t scale);

/**
 * @brief Performs a fully connected (linear) layer without activation.
 */
void linear(uint32_t input_size, uint32_t output_size, uint8_t* activation, uint8_t* output, int8_t* filter,
            int32_t* bias, uint32_t scale);

/**
 * @brief Quantizes floating-point values into 8-bit integers.
 */
void quantize(float* input_in_DRAM, uint8_t* output_in_DRAM, uint32_t size, uint32_t scale);

/**
 * @brief Dequantizes 8-bit integers back to floating-point values.
 */
void dequantize(uint8_t* input_in_DRAM, float* output_in_DRAM, uint32_t size, uint32_t scale);

#endif  // HARDWARE_CPU_H

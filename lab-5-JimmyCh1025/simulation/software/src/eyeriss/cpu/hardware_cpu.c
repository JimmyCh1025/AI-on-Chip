#include "hardware_cpu.h"

#include <stdio.h>

void conv_maxpooling(uint32_t input_C, uint32_t input_H, uint32_t input_W, uint8_t* activation, uint32_t filter_N,
                     uint32_t filter_C, uint32_t filter_H, uint32_t filter_W, int8_t* filter, int32_t* bias,
                     uint32_t padding, uint8_t* output, uint32_t scale) {
  //! hint>>
  if (padding != 1) {
    printf("Error: padding must be 1.\n");
    exit(1);
  }
  uint32_t pad_offset = 1;
  uint32_t padded_H   = input_H + 2 * pad_offset;
  uint32_t padded_W   = input_W + 2 * pad_offset;

  uint32_t conv_H = input_H;
  uint32_t conv_W = input_W;
  // output size
  uint32_t output_H = conv_H / 2;
  uint32_t output_W = conv_W / 2;

  // malloc padding activation space
  uint32_t act_pad_size      = input_C * padded_H * padded_W;
  uint8_t* activation_padded = (uint8_t*)malloc(act_pad_size * sizeof(uint8_t));
  if (!activation_padded) { exit(1); }
  memset(activation_padded, 128, act_pad_size * sizeof(uint8_t));
  // padding activation into malloc space
  for (uint32_t c = 0; c < input_C; c++) {
    uint32_t pad_channel_offset = c * padded_H * padded_W;
    uint32_t act_channel_offset = c * input_H * input_W;
    for (uint32_t h = 0; h < input_H; h++) {
      memcpy(activation_padded + pad_channel_offset + (h + pad_offset) * padded_W + pad_offset,
             activation + act_channel_offset + h * input_W, input_W * sizeof(uint8_t));
    }
  }

  uint32_t act_channel_stride = padded_H * padded_W;
  uint32_t filter_unit        = filter_C * filter_H * filter_W;

  int8_t* filter_buffer = (int8_t*)malloc(filter_unit * sizeof(int8_t));
  if (!filter_buffer) { exit(1); }
  for (uint32_t n = 0; n < filter_N; n++) {
    uint32_t base_filter_index = n * filter_unit;
    for (uint32_t c = 0; c < filter_C; c++) {
      uint32_t filter_channel_offset = c * filter_H * filter_W;
      for (uint32_t fh = 0; fh < filter_H; fh++) {
        memcpy(filter_buffer + filter_channel_offset + fh * filter_W,
               filter + base_filter_index + filter_channel_offset + fh * filter_W, filter_W * sizeof(int8_t));
      }
    }

    for (uint32_t no_pool_h = 0; no_pool_h < padded_H; no_pool_h += 2) {
      for (uint32_t no_pool_w = 0; no_pool_w < padded_W; no_pool_w += 2) {
        int32_t sum00 = bias[n];
        int32_t sum01 = bias[n];
        int32_t sum10 = bias[n];
        int32_t sum11 = bias[n];
        // MAC the weigt and activation
        for (uint32_t c = 0; c < filter_C; c++) {
          for (uint32_t fh = 0; fh < filter_H; fh++) {
            uint32_t act_base = c * act_channel_stride;

            uint8_t* act_ptr00 = activation_padded + act_base + (no_pool_h + fh) * padded_W + no_pool_w;
            uint8_t* act_ptr01 = activation_padded + act_base + (no_pool_h + fh) * padded_W + no_pool_w + 1;
            uint8_t* act_ptr10 = activation_padded + act_base + (no_pool_h + 1 + fh) * padded_W + no_pool_w;
            uint8_t* act_ptr11 = activation_padded + act_base + (no_pool_h + 1 + fh) * padded_W + no_pool_w + 1;

            uint32_t filt_base = c * filter_H * filter_W;
            int8_t*  filt_ptr  = filter_buffer + filt_base + fh * filter_W;

            for (uint32_t fw = 0; fw < filter_W; fw++) {
              int8_t filter_val = filt_ptr[fw];
              sum00 += (((int32_t)act_ptr00[fw]) - 128) * filter_val;
              sum01 += (((int32_t)act_ptr01[fw]) - 128) * filter_val;
              sum10 += (((int32_t)act_ptr10[fw]) - 128) * filter_val;
              sum11 += (((int32_t)act_ptr11[fw]) - 128) * filter_val;
            }
          }
        }
        uint8_t conv00 = requant(relu(sum00), scale);
        uint8_t conv01 = requant(relu(sum01), scale);
        uint8_t conv10 = requant(relu(sum10), scale);
        uint8_t conv11 = requant(relu(sum11), scale);
        // 2x2 pooling
        uint8_t max_val = conv00;
        if (conv01 > max_val) max_val = conv01;
        if (conv10 > max_val) max_val = conv10;
        if (conv11 > max_val) max_val = conv11;

        uint32_t pooled_row                                                    = no_pool_h / 2;
        uint32_t pooled_col                                                    = no_pool_w / 2;
        output[n * (output_H * output_W) + pooled_row * output_W + pooled_col] = max_val;
      }
    }
  }
  free(filter_buffer);
  free(activation_padded);
  //! hint<<
};

void conv(uint32_t input_C, uint32_t input_H, uint32_t input_W, uint8_t* activation, uint32_t filter_N,
          uint32_t filter_C, uint32_t filter_H, uint32_t filter_W, int8_t* filter, int32_t* bias, uint32_t padding,
          uint8_t* output, uint32_t scale) {
  //! hint>>
  uint32_t padded_H          = input_H + 2;
  uint32_t padded_W          = input_W + 2;
  uint32_t output_H          = input_H;
  uint32_t output_W          = input_W;
  uint32_t act_pad_size      = input_C * padded_H * padded_W;
  uint8_t* activation_padded = (uint8_t*)malloc(act_pad_size * sizeof(uint8_t));
  if (!activation_padded) { exit(1); }

  memset(activation_padded, 128, act_pad_size * sizeof(uint8_t));

  for (uint32_t c = 0; c < input_C; c++) {
    uint32_t pad_channel_offset = c * padded_H * padded_W;
    uint32_t act_channel_offset = c * input_H * input_W;
    for (uint32_t h = 0; h < input_H; h++) {
      memcpy(activation_padded + pad_channel_offset + (h + 1) * padded_W + 1,
             activation + act_channel_offset + h * input_W, input_W * sizeof(uint8_t));
    }
  }

  uint32_t act_channel_stride = padded_H * padded_W;
  uint32_t out_channel_stride = output_H * output_W;
  uint32_t filter_unit        = filter_C * filter_H * filter_W;

  int8_t* filter_buffer = (int8_t*)malloc(filter_unit * sizeof(int8_t));
  if (!filter_buffer) { exit(1); }
  for (uint32_t n = 0; n < filter_N; n++) {
    uint32_t base_filter_index = n * filter_unit;
    for (uint32_t c = 0; c < filter_C; c++) {
      uint32_t filter_channel_offset = c * filter_H * filter_W;
      for (uint32_t fh = 0; fh < filter_H; fh++) {
        memcpy(filter_buffer + filter_channel_offset + fh * filter_W,
               filter + base_filter_index + filter_channel_offset + fh * filter_W, filter_W * sizeof(int8_t));
      }
    }

    for (uint32_t output_idx_h = 0; output_idx_h < output_H; output_idx_h += 2) {
      for (uint32_t output_idx_w = 0; output_idx_w < output_W; output_idx_w += 2) {
        int32_t sum00 = bias[n];
        int32_t sum01 = bias[n];
        int32_t sum10 = bias[n];
        int32_t sum11 = bias[n];
        // MAC the weigt and activation
        for (uint32_t c = 0; c < filter_C; c++) {
          for (uint32_t fh = 0; fh < filter_H; fh++) {
            uint32_t act_base = c * act_channel_stride;

            uint8_t* act_ptr00 = activation_padded + act_base + (output_idx_h + fh) * padded_W + output_idx_w;
            uint8_t* act_ptr01 = activation_padded + act_base + (output_idx_h + fh) * padded_W + output_idx_w + 1;
            uint8_t* act_ptr10 = activation_padded + act_base + (output_idx_h + 1 + fh) * padded_W + output_idx_w;
            uint8_t* act_ptr11 = activation_padded + act_base + (output_idx_h + 1 + fh) * padded_W + output_idx_w + 1;

            uint32_t filt_base = c * filter_H * filter_W;
            int8_t*  filt_ptr  = filter_buffer + filt_base + fh * filter_W;

            for (uint32_t fw = 0; fw < filter_W; fw++) {
              int8_t filter_val = filt_ptr[fw];
              sum00 += (((int32_t)act_ptr00[fw]) - 128) * filter_val;
              sum01 += (((int32_t)act_ptr01[fw]) - 128) * filter_val;
              sum10 += (((int32_t)act_ptr10[fw]) - 128) * filter_val;
              sum11 += (((int32_t)act_ptr11[fw]) - 128) * filter_val;
            }
          }
        }
        uint8_t conv00 = requant(relu(sum00), scale);
        uint8_t conv01 = requant(relu(sum01), scale);
        uint8_t conv10 = requant(relu(sum10), scale);
        uint8_t conv11 = requant(relu(sum11), scale);

        output[n * (output_H * output_W) + output_idx_h * output_W + output_idx_w]           = conv00;
        output[n * (output_H * output_W) + output_idx_h * output_W + output_idx_w + 1]       = conv01;
        output[n * (output_H * output_W) + (output_idx_h + 1) * output_W + output_idx_w]     = conv10;
        output[n * (output_H * output_W) + (output_idx_h + 1) * output_W + output_idx_w + 1] = conv11;
      }
    }
  }
  free(filter_buffer);
  free(activation_padded);
  //! hint<<
};

void linear_relu(uint32_t input_size, uint32_t output_size, uint8_t* activation, uint8_t* output, int8_t* filter,
                 int32_t* bias, uint32_t scale) {
  //! hint>>
  int32_t* activation_buffer = (int32_t*)malloc(input_size * sizeof(int32_t));
  if (!activation_buffer) exit(1);

  for (uint32_t j = 0; j < input_size; j++) { activation_buffer[j] = (int32_t)activation[j] - 128; }

  for (uint32_t i = 0; i < output_size; i++) {
    int32_t sum = bias[i];

    const int8_t* f = filter + i * input_size;

    uint32_t j = 0;

    // loop unrolling with 4
    for (; j + 3 < input_size; j += 4) {
      sum += activation_buffer[j] * f[j];
      sum += activation_buffer[j + 1] * f[j + 1];
      sum += activation_buffer[j + 2] * f[j + 2];
      sum += activation_buffer[j + 3] * f[j + 3];
    }
    // handle element which %4 !=0
    for (; j < input_size; j++) { sum += activation_buffer[j] * f[j]; }
    output[i] = requant(relu(sum), scale);
  }
  free(activation_buffer);
  //! hint<<
};

void linear(uint32_t input_size, uint32_t output_size, uint8_t* activation, uint8_t* output, int8_t* filter,
            int32_t* bias, uint32_t scale) {
  //! hint>>
  int32_t* activation_buffer = (int32_t*)malloc(input_size * sizeof(int32_t));

  if (!activation_buffer) exit(1);

  for (uint32_t j = 0; j < input_size; j++) { activation_buffer[j] = (int32_t)activation[j] - 128; }

  for (uint32_t i = 0; i < output_size; i++) {
    int32_t sum = bias[i];

    const int8_t* f = filter + i * input_size;

    uint32_t j = 0;

    // loop unrolling with 4
    for (; j + 3 < input_size; j += 4) {
      sum += activation_buffer[j] * f[j];
      sum += activation_buffer[j + 1] * f[j + 1];
      sum += activation_buffer[j + 2] * f[j + 2];
      sum += activation_buffer[j + 3] * f[j + 3];
    }
    // handle element which %4 !=0
    for (; j < input_size; j++) { sum += activation_buffer[j] * f[j]; }
    output[i] = requant(sum, scale);
  }
  free(activation_buffer);
  //! hint<<
};

void quantize(float* input_in_DRAM, uint8_t* output_in_DRAM, uint32_t size, uint32_t scale) {
  float fp_scale = 1;
  for (uint32_t i = 0; i < scale; i++) { fp_scale *= 2; }
  for (uint32_t i = 0; i < size; i++) {
    float   t    = input_in_DRAM[i] * fp_scale;
    int32_t temp = (int32_t)t + 128;
    // clamp to 0 ~ 255
    if (temp < 0) {
      output_in_DRAM[i] = 0;
    } else if (temp > 255)
      output_in_DRAM[i] = 255;
    else
      output_in_DRAM[i] = (uint8_t)temp;
  }
};

void dequantize(uint8_t* input_in_DRAM, float* output_in_DRAM, uint32_t size, uint32_t scale) {
  float fp_scale = 1;
  for (uint32_t i = 0; i < scale; i++) { fp_scale *= 2; }
  for (uint32_t i = 0; i < size; i++) {
    float temp            = *(input_in_DRAM + i) - 128;
    *(output_in_DRAM + i) = temp / fp_scale;
  }
};

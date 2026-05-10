#include "kernel_cpu.h"

void conv_maxpooling(uint32_t input_C, uint32_t input_H, uint32_t input_W,
                     uint8_t* activation, uint32_t filter_N, uint32_t filter_C,
                     uint32_t filter_H, uint32_t filter_W, int8_t* filter,
                     int32_t* bias, uint32_t padding, uint8_t* output,
                     uint32_t scale) {
    const int32_t H = (int32_t)input_H;
    const int32_t W = (int32_t)input_W;
    const int32_t N = (int32_t)filter_N;
    const int32_t C = (int32_t)filter_C;
    const int32_t FH = (int32_t)filter_H;
    const int32_t FW = (int32_t)filter_W;
    const int32_t H_lim = H + 1;
    const int32_t W_lim = W + 1;
    const int32_t out_H = H >> 1;
    const int32_t out_W = W >> 1;

    for (int32_t n = 0; n < N; n++) {
        for (int32_t h = 0; h < out_H; h++) {
            for (int32_t w = 0; w < out_W; w++) {
                int32_t temp_out = INT32_MIN;
                for (int32_t m_h = 0; m_h < 2; m_h++) {
                    for (int32_t m_w = 0; m_w < 2; m_w++) {
                        int32_t temp = bias[n];
                        int32_t origin_h = h * 2 + m_h;
                        int32_t origin_w = w * 2 + m_w;
                        for (int32_t c = 0; c < C; c++) {
                            for (int32_t fh = 0; fh < FH; fh++) {
                                for (int32_t fw = 0; fw < FW; fw++) {
                                    int32_t in_h = origin_h + fh;
                                    int32_t in_w = origin_w + fw;
                                    if (in_h != 0 && in_h < H_lim &&
                                        in_w != 0 && in_w < W_lim) {
                                        int32_t activation_index =
                                            c * H * W + (in_h - 1) * W +
                                            (in_w - 1);
                                        int32_t filter_index = n * C * FH * FW +
                                                               c * FH * FW +
                                                               fh * FW + fw;
                                        int32_t activation_val =
                                            activation[activation_index] - 128;
                                        int32_t weight_val =
                                            filter[filter_index];
                                        temp += activation_val * weight_val;
                                    }
                                }
                            }
                        }
                        if (temp_out < temp) temp_out = temp;
                    }
                }
                uint32_t temp_out_relu = relu(temp_out);
                uint8_t temp_out_final = requant(temp_out_relu, scale);
                output[n * out_H * out_W + h * out_W + w] = temp_out_final;
            }
        }
    }
};

void conv(uint32_t input_C, uint32_t input_H, uint32_t input_W,
          uint8_t* activation, uint32_t filter_N, uint32_t filter_C,
          uint32_t filter_H, uint32_t filter_W, int8_t* filter, int32_t* bias,
          uint32_t padding, uint8_t* output, uint32_t scale) {
    const int32_t H = (int32_t)input_H;
    const int32_t W = (int32_t)input_W;
    const int32_t N = (int32_t)filter_N;
    const int32_t C = (int32_t)filter_C;
    const int32_t FH = (int32_t)filter_H;
    const int32_t FW = (int32_t)filter_W;
    const int32_t PAD = (int32_t)padding;
    const int32_t H_lim = H + PAD;
    const int32_t W_lim = W + PAD;

    for (int32_t n = 0; n < N; n++) {
        for (int32_t h = 0; h < H; h++) {
            for (int32_t w = 0; w < W; w++) {
                int32_t temp = bias[n];
                for (int32_t c = 0; c < C; c++) {
                    for (int32_t fh = 0; fh < FH; fh++) {
                        int32_t in_h = h + fh;
                        if (in_h == 0 || in_h >= H_lim) continue;
                        for (int32_t fw = 0; fw < FW; fw++) {
                            int32_t in_w = w + fw;
                            if (in_w == 0 || in_w >= W_lim) continue;
                            int32_t activation_index =
                                c * H * W + (in_h - 1) * W + (in_w - 1);
                            int32_t filter_index =
                                n * C * FH * FW + c * FH * FW + fh * FW + fw;
                            int32_t activation_val =
                                (int32_t)activation[activation_index] - 128;
                            int32_t weight_val = (int32_t)filter[filter_index];
                            temp += activation_val * weight_val;
                        }
                    }
                }
                uint32_t temp_relu = relu(temp);
                uint8_t temp_out = requant(temp_relu, scale);
                output[n * H * W + h * W + w] = temp_out;
            }
        }
    }
};

void linear_relu(uint32_t input_size, uint32_t output_size, uint8_t* activation,
                 uint8_t* output, int8_t* filter, int32_t* bias,
                 uint32_t scale) {
    for (uint32_t i = 0; i < output_size; i++) {
        int32_t temp = bias[i];
        int32_t activation_val;
        int32_t weight_val;
        for (uint32_t j = 0; j < input_size; j++) {
            activation_val = activation[j] - 128;
            weight_val = filter[i * input_size + j];
            temp += activation_val * weight_val;
        }
        uint32_t temp_relu = relu(temp);
        uint8_t temp_out = requant(temp_relu, scale);
        output[i] = temp_out;
    }
};

void linear(uint32_t input_size, uint32_t output_size, uint8_t* activation,
            uint8_t* output, int8_t* filter, int32_t* bias, uint32_t scale) {
    for (uint32_t i = 0; i < output_size; i++) {
        int32_t temp = bias[i];
        int32_t activation_val;
        int32_t weight_val;
        for (uint32_t j = 0; j < input_size; j++) {
            activation_val = activation[j] - 128;
            weight_val = filter[i * input_size + j];
            temp += activation_val * weight_val;
        }
        int8_t temp_shift = (temp >> scale);
        uint8_t temp_out = temp_shift + 128;
        output[i] = temp_out;
    }
};

#include "kernel_cpu.h"

void conv_maxpooling(uint32_t input_C, uint32_t input_H, uint32_t input_W,
                     uint8_t* activation, uint32_t filter_N, uint32_t filter_C,
                     uint32_t filter_H, uint32_t filter_W, int8_t* filter,
                     int32_t* bias, uint32_t padding, uint8_t* output,
                     uint32_t scale, void* scratch) {

    const int32_t H = (int32_t)input_H;
    const int32_t W = (int32_t)input_W;
    const int32_t N = (int32_t)filter_N;
    const int32_t C = (int32_t)filter_C;
    const int32_t FH = (int32_t)filter_H;
    const int32_t FW = (int32_t)filter_W;
    const int32_t out_H = H >> 1;
    const int32_t out_W = W >> 1;
    const int32_t PAD = (int32_t)padding;

    //  Loop-Invariant Code Motion
    const int32_t ifmap_stride = H * W;
    const int32_t filter_stride = FH * FW;
    const int32_t filter_batch_stride = C * filter_stride;
    const int32_t output_stride = out_H * out_W;
    
    int8_t* pad_buf = (int8_t*)scratch;
    const int32_t pad_H = H + 2 * PAD;
    const int32_t pad_W = W + 2 * PAD;
    const int32_t pad_stride = pad_H * pad_W;
    const int32_t total_pad_size = C * pad_H * pad_W;

    int32_t n = 0;

    memset(pad_buf, 0, total_pad_size);

    // use scratch
    for (int32_t c = 0; c < C; c++) {
        for (int32_t h = 0; h < H; h++) {
            for (int32_t w = 0; w < W; w++) {
                int32_t pad_idx = c * pad_H * pad_W + (h + PAD) * pad_W + (w + PAD);
                int32_t act_idx = c * H * W + h * W + w;
                pad_buf[pad_idx] = (int8_t)((int32_t)activation[act_idx] - 128);
            }
        }
    }

    // filter reuse
    for ( ; n <= (N-4); n+=4) 
    {
        for (int32_t h = 0; h < out_H; h++) 
        {
            for (int32_t w = 0; w < out_W; w++) 
            {
                int32_t temp_out0 = INT32_MIN;
                int32_t temp_out1 = INT32_MIN;
                int32_t temp_out2 = INT32_MIN;
                int32_t temp_out3 = INT32_MIN;

                for (int32_t m_h = 0; m_h < 2; m_h++) 
                {
                    // shift
                    int32_t origin_h = (h << 1) + m_h;

                    for (int32_t m_w = 0; m_w < 2; m_w++) 
                    {
                        int32_t acc0 = bias[n  ];
                        int32_t acc1 = bias[n+1];
                        int32_t acc2 = bias[n+2];
                        int32_t acc3 = bias[n+3];
                        
                        int8_t *weight_val0 = filter + ((n  ) * filter_batch_stride);
                        int8_t *weight_val1 = filter + ((n+1) * filter_batch_stride);
                        int8_t *weight_val2 = filter + ((n+2) * filter_batch_stride);
                        int8_t *weight_val3 = filter + ((n+3) * filter_batch_stride);

                        // shift
                        int32_t origin_w = (w << 1) + m_w;

                        for (int32_t c = 0; c < C; c++) 
                        {
                            int32_t filter_channel = c * filter_stride;
                            int32_t pad_channel_base = c * pad_stride;
                            for (int32_t fh = 0; fh < FH; fh++) 
                            {
                                int32_t pad_h_idx = origin_h + fh;

                                for (int32_t fw = 0; fw < FW; fw++) 
                                {
                                    int32_t pad_w_idx = origin_w + fw;

                                    int32_t filter_index = filter_channel + fh * FW + fw;
                                    int32_t pad_index = pad_channel_base + pad_h_idx * pad_W + pad_w_idx;
                                    
                                    int32_t activation_val = pad_buf[pad_index];

                                    acc0 += weight_val0[filter_index] * activation_val;
                                    acc1 += weight_val1[filter_index] * activation_val;
                                    acc2 += weight_val2[filter_index] * activation_val;
                                    acc3 += weight_val3[filter_index] * activation_val;
                                }
                            }
                        }
                        if (temp_out0 < acc0) temp_out0 = acc0;
                        if (temp_out1 < acc1) temp_out1 = acc1;
                        if (temp_out2 < acc2) temp_out2 = acc2;
                        if (temp_out3 < acc3) temp_out3 = acc3;
                    }
                }

                int32_t output_idx = n * output_stride + h * out_W + w;
                output[output_idx                  ] = requant(relu(temp_out0), scale);
                output[output_idx +   output_stride] = requant(relu(temp_out1), scale);
                output[output_idx + 2*output_stride] = requant(relu(temp_out2), scale);
                output[output_idx + 3*output_stride] = requant(relu(temp_out3), scale);
            }
        }
    }

    for ( ; n < N; ++n) 
    {
        for (int32_t h = 0; h < out_H; h++) 
        {
            for (int32_t w = 0; w < out_W; w++) 
            {
                int32_t temp_out = INT32_MIN;

                for (int32_t m_h = 0; m_h < 2; m_h++) 
                {
                    // shift
                    int32_t origin_h = (h << 1) + m_h;

                    for (int32_t m_w = 0; m_w < 2; m_w++) 
                    {
                        int32_t temp = bias[n];
                        int8_t *weight_val = filter + (n * filter_batch_stride);
                        int32_t origin_w = (w << 1) + m_w;
                        
                        for (int32_t c = 0; c < C; c++) 
                        {
                            int32_t filter_channel = c * filter_stride;
                            int32_t pad_channel_base = c * pad_stride;

                            for (int32_t fh = 0; fh < FH; fh++) 
                            {
                                int32_t pad_h_idx = origin_h + fh;

                                for (int32_t fw = 0; fw < FW; fw++) 
                                {
                                    int32_t pad_w_idx = origin_w + fw;

                                    int32_t filter_index = filter_channel + fh * FW + fw;
                                    int32_t pad_index = pad_channel_base + pad_h_idx * pad_W + pad_w_idx;
                                    
                                    temp += weight_val[filter_index] * pad_buf[pad_index];
                                }
                            }
                        }
                        if (temp_out < temp) temp_out = temp;
                    }
                }

                output[n * output_stride + h * out_W + w] = requant(relu(temp_out), scale);
            }
        }
    }
};

void conv(uint32_t input_C, uint32_t input_H, uint32_t input_W,
          uint8_t* activation, uint32_t filter_N, uint32_t filter_C,
          uint32_t filter_H, uint32_t filter_W, int8_t* filter, int32_t* bias,
          uint32_t padding, uint8_t* output, uint32_t scale, void* scratch) {

    const int32_t H = (int32_t)input_H;
    const int32_t W = (int32_t)input_W;
    const int32_t N = (int32_t)filter_N;
    const int32_t C = (int32_t)filter_C;
    const int32_t FH = (int32_t)filter_H;
    const int32_t FW = (int32_t)filter_W;
    const int32_t PAD = (int32_t)padding;

    //  Loop-Invariant Code Motion
    const int32_t ifmap_stride = H * W;
    const int32_t filter_stride = FH * FW;
    const int32_t filter_batch_stride = C * FH * FW;


    int32_t n = 0;

    // filter reuse
    for ( ; n <= (N-4); n+=4) 
    {
        for (int32_t h = 0; h < H; h++) 
        {
            for (int32_t w = 0; w < W; w++) 
            {
                int32_t acc0 = bias[n  ];
                int32_t acc1 = bias[n+1];
                int32_t acc2 = bias[n+2];
                int32_t acc3 = bias[n+3];

                int8_t *weight_val0 = filter + ((n  ) * filter_batch_stride);
                int8_t *weight_val1 = filter + ((n+1) * filter_batch_stride);
                int8_t *weight_val2 = filter + ((n+2) * filter_batch_stride);
                int8_t *weight_val3 = filter + ((n+3) * filter_batch_stride);

                for (int32_t c = 0; c < C; c++) 
                {
                    for (int32_t fh = 0; fh < FH; ++fh) 
                    {
                        int32_t in_h = h - PAD + fh;
                        if (in_h < 0 || in_h >= H) continue;

                        for (int32_t fw = 0; fw < FW; ++fw) 
                        {
                            int32_t in_w = w - PAD + fw;
                            if (in_w < 0 || in_w >= W) continue;
                            int32_t activation_index = c * ifmap_stride + in_h * W + in_w;
                            int32_t activation_val = activation[activation_index] - 128;
                            int32_t filter_index = c * filter_stride + fh * FW + fw;
                            acc0 += weight_val0[filter_index] * activation_val;
                            acc1 += weight_val1[filter_index] * activation_val;
                            acc2 += weight_val2[filter_index] * activation_val;
                            acc3 += weight_val3[filter_index] * activation_val;
               
                        }
                    }
                }
                int32_t output_idx = n * ifmap_stride + h * W + w;
                output[output_idx                 ] = requant(relu(acc0), scale);
                output[output_idx +   ifmap_stride] = requant(relu(acc1), scale);
                output[output_idx + 2*ifmap_stride] = requant(relu(acc2), scale);
                output[output_idx + 3*ifmap_stride] = requant(relu(acc3), scale);
            }
        }
    }

    for ( ; n < N; ++n) 
    {
        for (int32_t h = 0; h < H; h++) 
        {
            for (int32_t w = 0; w < W; w++) 
            {
                int32_t temp = bias[n];
                int8_t *weight_val = filter + ((n  ) * filter_batch_stride);

                for (int32_t c = 0; c < C; c++) 
                {
                    for (int32_t fh = 0; fh < FH; fh++) 
                    {
                        int32_t in_h = h - PAD + fh;
                        if (in_h < 0 || in_h >= H) continue;
                    
                        for (int32_t fw = 0 ; fw < FW ; ++fw) 
                        {
                            int32_t in_w = w - PAD + fw;
                            if (in_w < 0 || in_w >= W) continue;

                            int32_t activation_index = c * ifmap_stride + in_h * W + in_w;
                            int32_t activation_val = (int32_t)activation[activation_index] - 128;
                            int32_t filter_index = c * filter_stride + fh * FW + fw;

                            temp += activation_val * weight_val[filter_index];
                        }
                    }
                }
                output[n * ifmap_stride + h * W + w] = requant(relu(temp), scale);
            }
        }
    }
};

void linear_relu(uint32_t input_size, uint32_t output_size, uint8_t* activation,
                 uint8_t* output, int8_t* filter, int32_t* bias, uint32_t scale,
                 void* scratch) {

    uint32_t i = 0;
    for ( ; i < (output_size-3); i+=4) 
    {
        int32_t acc0 = bias[i  ];
        int32_t acc1 = bias[i+1];
        int32_t acc2 = bias[i+2];
        int32_t acc3 = bias[i+3];

        int8_t *weight_val0 = filter + i * input_size;
        int8_t *weight_val1 = filter + (i+1) * input_size;
        int8_t *weight_val2 = filter + (i+2) * input_size;
        int8_t *weight_val3 = filter + (i+3) * input_size;

        uint32_t j = 0;

        for (; j < (input_size-3); j+=4) 
        {
            int32_t activation_val0 = (int32_t)activation[j  ] - 128;
            int32_t activation_val1 = (int32_t)activation[j+1] - 128;
            int32_t activation_val2 = (int32_t)activation[j+2] - 128;
            int32_t activation_val3 = (int32_t)activation[j+3] - 128;

            acc0 += weight_val0[j  ] * activation_val0;
            acc1 += weight_val1[j  ] * activation_val0;
            acc2 += weight_val2[j  ] * activation_val0;
            acc3 += weight_val3[j  ] * activation_val0;

            acc0 += weight_val0[j+1] * activation_val1;
            acc1 += weight_val1[j+1] * activation_val1;
            acc2 += weight_val2[j+1] * activation_val1;
            acc3 += weight_val3[j+1] * activation_val1;

            acc0 += weight_val0[j+2] * activation_val2;
            acc1 += weight_val1[j+2] * activation_val2;
            acc2 += weight_val2[j+2] * activation_val2;
            acc3 += weight_val3[j+2] * activation_val2;

            acc0 += weight_val0[j+3] * activation_val3;
            acc1 += weight_val1[j+3] * activation_val3;
            acc2 += weight_val2[j+3] * activation_val3;
            acc3 += weight_val3[j+3] * activation_val3;
        }

        // remainder input
        for ( ; j < input_size ; ++j)
        {
            int32_t activation_val = activation[j] - 128;
            acc0 += weight_val0[j] * activation_val;
            acc1 += weight_val1[j] * activation_val;
            acc2 += weight_val2[j] * activation_val;
            acc3 += weight_val3[j] * activation_val;
        }

        output[i  ] = requant(relu(acc0), scale);
        output[i+1] = requant(relu(acc1), scale);
        output[i+2] = requant(relu(acc2), scale);
        output[i+3] = requant(relu(acc3), scale);
    }

    // remainder output
    for ( ; i < output_size ; ++i)
    {
        int32_t temp = bias[i];
        int8_t *weight_val = filter + i * input_size;

        for (uint32_t j = 0; j < input_size; ++j) 
        {
            int32_t activation_val = activation[j] - 128;
            temp += activation_val * weight_val[j];
        }

        output[i] = requant(relu(temp), scale);
    }
};

void linear(uint32_t input_size, uint32_t output_size, uint8_t* activation,
            uint8_t* output, int8_t* filter, int32_t* bias, uint32_t scale,
            void* scratch) {
 
    uint32_t i = 0;
    for ( ; i < (output_size-3); i+=4) 
    {
        int32_t acc0 = bias[i  ];
        int32_t acc1 = bias[i+1];
        int32_t acc2 = bias[i+2];
        int32_t acc3 = bias[i+3];

        int8_t *weight_val0 = filter + i * input_size;
        int8_t *weight_val1 = filter + (i+1) * input_size;
        int8_t *weight_val2 = filter + (i+2) * input_size;
        int8_t *weight_val3 = filter + (i+3) * input_size;

        uint32_t j = 0;

        for (; j < (input_size-3); j+=4) 
        {
            int32_t activation_val0 = activation[j  ] - 128;
            int32_t activation_val1 = activation[j+1] - 128;
            int32_t activation_val2 = activation[j+2] - 128;
            int32_t activation_val3 = activation[j+3] - 128;

            acc0 += weight_val0[j  ] * activation_val0;
            acc1 += weight_val1[j  ] * activation_val0;
            acc2 += weight_val2[j  ] * activation_val0;
            acc3 += weight_val3[j  ] * activation_val0;

            acc0 += weight_val0[j+1] * activation_val1;
            acc1 += weight_val1[j+1] * activation_val1;
            acc2 += weight_val2[j+1] * activation_val1;
            acc3 += weight_val3[j+1] * activation_val1;

            acc0 += weight_val0[j+2] * activation_val2;
            acc1 += weight_val1[j+2] * activation_val2;
            acc2 += weight_val2[j+2] * activation_val2;
            acc3 += weight_val3[j+2] * activation_val2;

            acc0 += weight_val0[j+3] * activation_val3;
            acc1 += weight_val1[j+3] * activation_val3;
            acc2 += weight_val2[j+3] * activation_val3;
            acc3 += weight_val3[j+3] * activation_val3;
        }

        // remainder input
        for ( ; j < input_size ; ++j)
        {
            int32_t activation_val = activation[j] - 128;
            acc0 += weight_val0[j] * activation_val;
            acc1 += weight_val1[j] * activation_val;
            acc2 += weight_val2[j] * activation_val;
            acc3 += weight_val3[j] * activation_val;
        }

        output[i  ] = (uint8_t)((int8_t)(acc0 >> scale) + 128);
        output[i+1] = (uint8_t)((int8_t)(acc1 >> scale) + 128);
        output[i+2] = (uint8_t)((int8_t)(acc2 >> scale) + 128);
        output[i+3] = (uint8_t)((int8_t)(acc3 >> scale) + 128);
    }

    // remainder output
    for ( ; i < output_size ; ++i)
    {
        int32_t temp = bias[i];
        int8_t *weight_val = filter + i * input_size;

        for (uint32_t j = 0; j < input_size; ++j) 
        {
            int32_t activation_val = activation[j] - 128;
            temp += activation_val * weight_val[j];
        }

        output[i] = (uint8_t)((int8_t)(temp >> scale) + 128);
    }
};

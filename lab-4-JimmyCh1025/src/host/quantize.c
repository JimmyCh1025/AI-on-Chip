#include "quantize.h"

#include <math.h>

/* activation: uint8, zero_point = 128 */

void quantize(const float* input, uint8_t* output, uint32_t size,
              uint32_t scale) {
    float fp_scale = ldexpf(1.0f, (int)scale); /* 2^scale */
    for (uint32_t i = 0; i < size; i++) {
        int32_t temp = (int32_t)roundf(input[i] * fp_scale) + 128;
        if (temp < 0)
            output[i] = 0;
        else if (temp > 255)
            output[i] = 255;
        else
            output[i] = (uint8_t)temp;
    }
}

void dequantize(const uint8_t* input, float* output, uint32_t size,
                uint32_t scale) {
    float fp_scale = ldexpf(1.0f, (int)scale);
    for (uint32_t i = 0; i < size; i++) {
        output[i] = ((float)input[i] - 128.0f) / fp_scale;
    }
}

/* weight: int8, symmetric, no zero-point */

void quantize_weights(const float* input, int8_t* output, uint32_t size,
                      uint32_t scale) {
    float fp_scale = ldexpf(1.0f, (int)scale);
    for (uint32_t i = 0; i < size; i++) {
        int32_t temp = (int32_t)roundf(input[i] * fp_scale);
        if (temp < -128)
            output[i] = -128;
        else if (temp > 127)
            output[i] = 127;
        else
            output[i] = (int8_t)temp;
    }
}

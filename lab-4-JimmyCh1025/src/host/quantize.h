#ifndef HOST_QUANTIZE_H
#define HOST_QUANTIZE_H

#include <stdint.h>

/* Host-side quantize/dequantize utilities (float <-> integer).
 * NOT compiled into the RV64 ELF; host-only preprocessing.
 *
 * Scheme:
 *   Activation (uint8): clamp(round(x * 2^scale) + 128, 0, 255)
 *   Weight     (int8):  clamp(round(x * 2^scale), -128, 127)
 */

/* Quantize float activations to uint8 (zero_point=128). */
void quantize(const float* input, uint8_t* output, uint32_t size,
              uint32_t scale);

/* Dequantize uint8 activations back to float. */
void dequantize(const uint8_t* input, float* output, uint32_t size,
                uint32_t scale);

/* Quantize float weights to int8 (symmetric). */
void quantize_weights(const float* input, int8_t* output, uint32_t size,
                      uint32_t scale);

#endif /* HOST_QUANTIZE_H */

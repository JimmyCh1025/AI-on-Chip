#ifndef RUNTIME_H
#define RUNTIME_H

/* default mapping parameters */
#define DEFAULT_m 64
#define DEFAULT_p 4
#define DEFAULT_q 4
#define DEFAULT_r 1
#define DEFAULT_t 2
#define DEFAULT_e 8

/* default dla info file name */
#define DLA_INFO_CSV "dla_info.csv"

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdio.h>

/*
    Parameter meaning:
    M : ofmap channel
    C : ifmap/filter channel
    H/W : ifmap size ,height/width
    R/S : filter size ,height/width
    PAD: padding number
    U: stride

    scale: The required number of bits to shift to the 'right',
           determined by combining the input scale, weight scale,
           and output scale, has been calculated to reduce the 32-bit
   computation back to 8-bit.

    Memory order: N C H W
*/

/**********************************
***                             ***
***           DLA API           ***
***                             ***
**********************************/

/**
 * @brief  Stops the Deep Learning Accelerator (DLA).
 */
void dla_stop();

/**
 * @brief  Initializes the Deep Learning Accelerator (DLA).
 *
 * This function initializes the DLA hardware and, if debugging is enabled
 * (`DLA_INFO`), resets runtime statistics and creates a CSV log file.
 *
 * @note   This function must be called before using any DLA operations.
 */
void dla_init();

/**
 * @brief  Finalizes and shuts down the Deep Learning Accelerator (DLA).
 */
void dla_final();

/**
 * @brief  Resets the runtime statistics of the Deep Learning Accelerator (DLA).
 */
void dla_reset_runtime_info();

/**
 * @brief  Creates a CSV file for DLA (Deep Learning Accelerator) information.
 *
 * This function creates a CSV file with the specified filename and writes the
 * header row containing column names.
 *
 * @param  filename  The name of the CSV file to be created.
 *
 * @note   If the file cannot be opened, an error message is printed to stderr.
 */
void create_dla_info_to_csv(const char* filename);

/**
 * @brief  Logs Deep Learning Accelerator (DLA) execution data to a CSV file.
 *
 * This function appends execution runtime information and operation parameters
 * to an existing CSV file. The logged data includes cycle count, elapsed time,
 * memory usage, and various mapping and shape parameters.
 *
 * @param  filename        The name of the CSV file to append data to.
 * @param  operation_name  The name of the operation being logged.
 * @param  m, e, p, q, r, t  Mapping parameters.
 * @param  PAD, U, R, S, C, M, W, H  Shape parameters.
 *
 * @note   The function assumes the CSV file already has a header row.
 * @warning If the file cannot be opened, data logging will fail silently.
 */
void dump_dla_info_to_csv(const char* filename, char* operation_name,
                          // mapping parameter
                          uint32_t m, uint32_t e, uint32_t p, uint32_t q, uint32_t r, uint32_t t,
                          // shape parameter
                          uint32_t PAD, uint32_t U, uint32_t R, uint32_t S, uint32_t C, uint32_t M, uint32_t W,
                          uint32_t H);

/**
 * @brief  Performs quantized 2D convolution, ReLU activation, and max pooling.
 *
 * This function configures the Deep Learning Accelerator (DLA) to execute a
 * quantized convolution operation followed by ReLU activation and max pooling.
 * It sets up memory addresses, mapping parameters, shape parameters, and runs
 * the computation on the accelerator.
 *
 * @param  input_in_DRAM   Pointer to input feature map stored in DRAM.
 * @param  filter_in_DRAM  Pointer to convolution filters stored in DRAM.
 * @param  opsum_in_DRAM   Pointer to output feature map storage in DRAM.
 * @param  bias            Pointer to bias values.
 * @param  ofmap_len       Length of the output feature map.
 * @param  ifmap_len       Length of the input feature map.
 * @param  filter_len      Length of the filter weights.
 *
 * @param  m, e, p, q, r, t  Mapping parameters for hardware execution.
 * @param  PAD, U, R, S, C, M, W, H  Shape parameters defining the convolution.
 * @param  scale           Scale factor for quantization (bit-shift operation).
 *
 * @return  Returns `0` on successful execution.
 *
 * @note   The function assumes that DLA hardware is properly initialized.
 * @warning This function waits for an interrupt (`wait_for_interrupt()`),
 *          which may block execution if not properly handled.
 */
int qconv2d_relu_maxpool(uint8_t* input_in_DRAM, int8_t* filter_in_DRAM, uint8_t* opsum_in_DRAM, int32_t* bias,
                         uint32_t ofmap_len, uint32_t ifmap_len, uint32_t filter_len,
                         // mapping parameter
                         uint32_t m, uint32_t e, uint32_t p, uint32_t q, uint32_t r, uint32_t t,
                         // shape parameter
                         uint32_t PAD, uint32_t U, uint32_t R, uint32_t S, uint32_t C, uint32_t M, uint32_t W,
                         uint32_t H, uint32_t scale);

/**
 * @brief  Performs quantized 2D convolution and ReLU activation.
 *
 * This function configures the Deep Learning Accelerator (DLA) to execute a
 * quantized convolution operation followed by ReLU activation.
 * It sets up memory addresses, mapping parameters, shape parameters, and runs
 * the computation on the accelerator.
 *
 * @param  input_in_DRAM   Pointer to input feature map stored in DRAM.
 * @param  filter_in_DRAM  Pointer to convolution filters stored in DRAM.
 * @param  opsum_in_DRAM   Pointer to output feature map storage in DRAM.
 * @param  bias            Pointer to bias values.
 * @param  ofmap_len       Length of the output feature map.
 * @param  ifmap_len       Length of the input feature map.
 * @param  filter_len      Length of the filter weights.
 *
 * @param  m, e, p, q, r, t  Mapping parameters for hardware execution.
 * @param  PAD, U, R, S, C, M, W, H  Shape parameters defining the convolution.
 * @param  scale           Scale factor for quantization (bit-shift operation).
 *
 * @return  Returns `0` on successful execution.
 *
 * @note   The function assumes that DLA hardware is properly initialized.
 * @warning This function waits for an interrupt (`wait_for_interrupt()`),
 *          which may block execution if not properly handled.
 */
int qconv2d_relu(uint8_t* input_in_DRAM, int8_t* filter_in_DRAM, uint8_t* opsum_in_DRAM, int32_t* bias,
                 uint32_t ofmap_len, uint32_t ifmap_len, uint32_t filter_len,
                 // mapping parameter
                 uint32_t m, uint32_t e, uint32_t p, uint32_t q, uint32_t r, uint32_t t,
                 // shape parameter
                 uint32_t PAD, uint32_t U, uint32_t R, uint32_t S, uint32_t C, uint32_t M, uint32_t W, uint32_t H,
                 uint32_t scale);

/**********************************
***                             ***
***           CPU API           ***
***                             ***
**********************************/

/**
 * @brief  Performs quantized 2D convolution, ReLU activation, and max pooling
 * on the CPU.
 *
 * This function executes a quantized 2D convolution followed by ReLU activation
 * and max pooling, entirely on the CPU.
 *
 * @param  input_in_DRAM   Pointer to input feature map stored in DRAM.
 * @param  filter_in_DRAM  Pointer to convolution filters stored in DRAM.
 * @param  opsum_in_DRAM   Pointer to output feature map storage in DRAM.
 * @param  bias            Pointer to bias values.
 * @param  ofmap_len       Length of the output feature map.
 * @param  ifmap_len       Length of the input feature map.
 * @param  filter_len      Length of the filter weights.
 * @param  PAD, U, R, S, C, M, W, H  Shape parameters defining the convolution.
 * @param  scale           Scale factor for quantization (bit-shift operation).
 *
 * @return  Returns `0` on successful execution.
 */
int qconv2d_relu_maxpool_cpu(uint8_t* input_in_DRAM, int8_t* filter_in_DRAM, uint8_t* opsum_in_DRAM, int32_t* bias,
                             uint32_t ofmap_len, uint32_t ifmap_len, uint32_t filter_len,
                             // shape parameter
                             uint32_t PAD, uint32_t U, uint32_t R, uint32_t S, uint32_t C, uint32_t M, uint32_t W,
                             uint32_t H, uint32_t scale);

/**
 * @brief  Performs quantized 2D convolution and ReLU activation on the CPU.
 *
 * Similar to `qconv2d_relu_maxpool_cpu()`, but without the max pooling step.
 *
 * @param  input_in_DRAM   Pointer to input feature map stored in DRAM.
 * @param  filter_in_DRAM  Pointer to convolution filters stored in DRAM.
 * @param  opsum_in_DRAM   Pointer to output feature map storage in DRAM.
 * @param  bias            Pointer to bias values.
 * @param  ofmap_len       Length of the output feature map.
 * @param  ifmap_len       Length of the input feature map.
 * @param  filter_len      Length of the filter weights.
 * @param  PAD, U, R, S, C, M, W, H  Shape parameters defining the convolution.
 * @param  scale           Scale factor for quantization (bit-shift operation).
 *
 * @return  Returns `0` on successful execution.
 */
int qconv2d_relu_cpu(uint8_t* input_in_DRAM, int8_t* filter_in_DRAM, uint8_t* opsum_in_DRAM, int32_t* bias,
                     uint32_t ofmap_len, uint32_t ifmap_len, uint32_t filter_len,
                     // shape parameter
                     uint32_t PAD, uint32_t U, uint32_t R, uint32_t S, uint32_t C, uint32_t M, uint32_t W, uint32_t H,
                     uint32_t scale);

/**
 * @brief  Performs a quantized linear transformation followed by ReLU
 * activation on the CPU.
 *
 * This function executes a fully connected (linear) layer with quantized
 * inputs, followed by ReLU activation.
 *
 * @param  input_in_DRAM   Pointer to input data in DRAM.
 * @param  filter_in_DRAM  Pointer to weight matrix in DRAM.
 * @param  opsum_in_DRAM   Pointer to output storage in DRAM.
 * @param  bias            Pointer to bias values.
 * @param  ofmap_len       Length of the output vector.
 * @param  ifmap_len       Length of the input vector.
 * @param  filter_len      Length of the weight matrix.
 * @param  scale           Scale factor for quantization (bit-shift operation).
 *
 * @return  Returns `0` on successful execution.
 */
int qlinear_relu_cpu(uint8_t* input_in_DRAM, int8_t* filter_in_DRAM, uint8_t* opsum_in_DRAM, int32_t* bias,
                     uint32_t ofmap_len, uint32_t ifmap_len, uint32_t filter_len, uint32_t scale);

/**
 * @brief  Performs a quantized linear transformation on the CPU.
 *
 * Similar to `qlinear_relu_cpu()`, but without the ReLU activation step.
 *
 * @param  input_in_DRAM   Pointer to input data in DRAM.
 * @param  filter_in_DRAM  Pointer to weight matrix in DRAM.
 * @param  opsum_in_DRAM   Pointer to output storage in DRAM.
 * @param  bias            Pointer to bias values.
 * @param  ofmap_len       Length of the output vector.
 * @param  ifmap_len       Length of the input vector.
 * @param  filter_len      Length of the weight matrix.
 * @param  scale           Scale factor for quantization (bit-shift operation).
 *
 * @return  Returns `0` on successful execution.
 */
int qlinear_cpu(uint8_t* input_in_DRAM, int8_t* filter_in_DRAM, uint8_t* opsum_in_DRAM, int32_t* bias,
                uint32_t ofmap_len, uint32_t ifmap_len, uint32_t filter_len, uint32_t scale);

/**
 * @brief  Quantizes floating-point input data to 8-bit integers.
 *
 * This function applies quantization to a floating-point input array,
 * converting it to an 8-bit integer representation based on a given scale
 * factor.
 *
 * @param  input_in_DRAM   Pointer to input floating-point data.
 * @param  output_in_DRAM  Pointer to quantized output (8-bit integers).
 * @param  size            Number of elements to quantize.
 * @param  scale           Scale factor for quantization (bit-shift operation).
 */
void quantize_cpu(float* input_in_DRAM, uint8_t* output_in_DRAM, uint32_t size, uint32_t scale);

/**
 * @brief  Dequantizes 8-bit integer data back to floating-point representation.
 *
 * This function converts quantized 8-bit integers back into floating-point
 * values based on a given scale factor.
 *
 * @param  input_in_DRAM   Pointer to quantized input data (8-bit integers).
 * @param  output_in_DRAM  Pointer to output floating-point data.
 * @param  size            Number of elements to dequantize.
 * @param  scale           Scale factor for dequantization (bit-shift
 * operation).
 */
void dequantize_cpu(uint8_t* input_in_DRAM, float* output_in_DRAM, uint32_t size, uint32_t scale);

#ifdef __cplusplus
}
#endif

#endif  // RUNTIME_H

// driver_dla.h — DLA register map, MMIO helpers, and driver prototypes.

#ifndef DRIVER_DLA_H
#define DRIVER_DLA_H

#include <stdbool.h>
#include <stdint.h>

#include "dla_hal.hpp"

/* ========================= DLA Register Base Address & Size
 * ========================= */
#define DLA_MMIO_BASE_ADDR 0x10040000  ///< Base address of the DLA MMIO registers.
#define DLA_MMIO_SIZE      0x1000      ///< Size of the DLA register memory map.

/* ========================= DLA Register Offsets ========================= */
#define DLA_ENABLE_OFFSET          0x0   ///< Offset for enabling/disabling the DLA.
#define DLA_MAPPING_PARAM_OFFSET   0x4   ///< Offset for setting mapping parameters.
#define DLA_SHAPE_PARAM1_OFFSET    0x8   ///< Offset for shape parameters (filter, channel).
#define DLA_SHAPE_PARAM2_OFFSET    0xc   ///< Offset for shape parameters (input size, padding).
#define DLA_IFMAP_ADDR_OFFSET      0x10  ///< Offset for input feature map address.
#define DLA_FILTER_ADDR_OFFSET     0x14  ///< Offset for filter weights address.
#define DLA_BIAS_ADDR_OFFSET       0x18  ///< Offset for bias values address.
#define DLA_OPSUM_ADDR_OFFSET      0x1c  ///< Offset for output sum buffer address.
#define DLA_GLB_FILTER_ADDR_OFFSET 0x20  ///< Offset for global filter weights address.
#define DLA_GLB_OFMAP_ADDR_OFFSET  0x24  ///< Offset for global output feature map address.
#define DLA_GLB_BIAS_ADDR_OFFSET   0x28  ///< Offset for global bias values address.
#define DLA_IFMAP_LEN_OFFSET       0x2c  ///< Offset for input activation length.
#define DLA_OFMAP_LEN_OFFSET       0x30  ///< Offset for output activation length.

#define DLA_UNDEFINED  ///< Placeholder for undefined registers.

/* ========================= Function Prototypes ========================= */

#ifdef __cplusplus
#include "dla_hal.hpp"
void set_dla_hal(DlaHAL* hal);
DlaHAL* get_dla_hal();
#endif /* __cplusplus */

/**
 * @brief Writes a value to a DLA register via memory-mapped I/O.
 * @param offset Register offset from `DLA_MMIO_BASE_ADDR`.
 * @param value The value to write.
 */
void reg_write(uint32_t offset, uint32_t value);

/**
 * @brief Enables the DLA and configures its operational mode.
 * @param scale_factor Bit-shift scale factor for quantization.
 * @param maxpool Enables max pooling (true/false).
 * @param relu Enables ReLU activation (true/false).
 * @param operation Specifies the type of operation.
 */
void set_enable(uint32_t scale_factor, bool maxpool, bool relu, bool operation);

/**
 * @brief Sets the mapping parameters for workload distribution.
 * @param m Tile size in the M-dimension.
 * @param e Expansion factor.
 * @param p Partition size.
 * @param q Stride in the Q-dimension.
 * @param r Stride in the R-dimension.
 * @param t Tile size in the T-dimension.
 */
void set_mapping_param(uint32_t m, uint32_t e, uint32_t p, uint32_t q,
                       uint32_t r, uint32_t t);

/**
 * @brief Configures shape parameters related to filter size, channels, and
 * padding.
 * @param PAD Padding size.
 * @param U Stride in the U-dimension.
 * @param R Filter height.
 * @param S Filter width.
 * @param C Input channels.
 * @param M Output channels.
 */
void set_shape_param1(uint32_t PAD, uint32_t U, uint32_t R, uint32_t S,
                      uint32_t C, uint32_t M);

/**
 * @brief Sets additional shape parameters including input width and height.
 * @param W Input width.
 * @param H Input height.
 * @param PAD Padding size.
 */
void set_shape_param2(uint32_t W, uint32_t H, uint32_t PAD);

/**
 * @brief Sets the address of the input feature map in DRAM.
 * @param addr Pointer to the input feature map data.
 */
void set_ifmap_addr(uint8_t* addr);

/**
 * @brief Sets the address of the filter weights in DRAM.
 * @param addr Pointer to the filter weight data.
 */
void set_filter_addr(int8_t* addr);

/**
 * @brief Sets the address of the bias values in DRAM.
 * @param addr Pointer to the bias data.
 */
void set_bias_addr(int32_t* addr);

/**
 * @brief Sets the address of the output sum buffer in DRAM.
 * @param addr Pointer to the output sum data.
 */
void set_opsum_addr(uint8_t* addr);

/**
 * @brief Sets the global address for filter weights in the DLA memory.
 * @param addr Memory address for filters.
 */
void set_glb_filter_addr(uint32_t addr);

/**
 * @brief Sets the global address for biases in the DLA memory.
 * @param addr Memory address for biases.
 */
void set_glb_bias_addr(uint32_t addr);

/**
 * @brief Sets the global address for output feature maps in the DLA memory.
 * @param addr Memory address for output feature maps.
 */
void set_glb_ofmap_addr(uint32_t addr);

/**
 * @brief Sets the input activation length (size of input data).
 * @param len The length of the input activation data.
 */
void set_input_activation_len(uint32_t len);

/**
 * @brief Sets the output activation length (size of output data).
 * @param len The length of the output activation data.
 */
void set_output_activation_len(uint32_t len);

#endif  // DRIVER_DLA_H
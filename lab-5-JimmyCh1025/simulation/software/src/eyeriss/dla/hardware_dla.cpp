/**
 * @file hardware_dla.cpp
 * @brief Provides an interface to configure and control a Deep Learning
 * Accelerator (DLA).
 *
 * This file implements functions to configure mapping parameters, shape
 * parameters, memory addresses, and enable various operations on the DLA. It
 * uses a Hardware Abstraction Layer (HAL) for memory-mapped I/O (MMIO)
 * interactions.
 */

#include "hardware_dla.h"

#include "hal.hpp"

/// @brief Global HAL instance for DLA interaction.
static HardwareAbstractionLayer hal(DLA_MMIO_BASE_ADDR,
                                    DLA_MMIO_SIZE);  // eyeriss device included

/* ========================= HAL Interface ========================= */
void wait_for_interrupt() { hal.wait_for_irq(); }
void hal_init() { hal.init(); }
void hal_final() { hal.final(); }

struct runtime_info get_runtime_info() { return hal.get_runtime_info(); }
void                reset_runtime_info() { hal.reset_runtime_info(); }

uint32_t mmu_map(void* host_ptr, size_t size) { return hal.mmu_map(host_ptr, size); }
void     mmu_unmap(uint32_t device_addr) { hal.mmu_unmap(device_addr); };

void reg_write(uint32_t offset, uint32_t value) { hal.memory_set(offset + DLA_MMIO_BASE_ADDR, value); }

/* ========================= DLA Configuration ========================= */
void set_enable(uint32_t scale_factor, bool maxpool, bool relu, bool operation) {
  uint32_t value;
  //! hint>>
  value = (scale_factor & 0x3f) << 4;
  value |= operation << 3;
  value |= relu << 2;
  value |= maxpool << 1;
  value |= 0x1;
  //! hint<<
  reg_write(DLA_ENABLE_OFFSET, value);
}

void set_mapping_param(uint32_t m, uint32_t e, uint32_t p, uint32_t q, uint32_t r, uint32_t t) {
  uint32_t value;
  //! hint>>
  value = (m & 0x3ff) << 16;
  value |= (e & 0xf) << 12;
  value |= (p & 0x7) << 9;
  value |= (q & 0x7) << 6;
  value |= (r & 0x7) << 3;
  value |= (t & 0x7);
  //! hint<<
  reg_write(DLA_MAPPING_PARAM_OFFSET, value);
}

void set_shape_param1(uint32_t PAD, uint32_t U, uint32_t R, uint32_t S, uint32_t C, uint32_t M) {
  uint32_t value;
  //! hint>>
  value = (PAD & 0x7) << 26;
  value |= (U & 0x3) << 24;
  value |= (R & 0x3) << 22;
  value |= (S & 0x3) << 20;
  value |= (C & 0x3ff) << 10;
  value |= (M & 0x3ff);
  //! hint<<
  reg_write(DLA_SHAPE_PARAM1_OFFSET, value);
}

void set_shape_param2(uint32_t W, uint32_t H, uint32_t PAD) {
  uint32_t value;
  //! hint>>
  uint32_t padding = (PAD & 0x7);
  value            = ((W + 2 * padding) & 0xff) << 8;
  value |= ((H + 2 * padding) & 0xff);
  //! hint<<
  reg_write(DLA_SHAPE_PARAM2_OFFSET, value);
}

void set_ifmap_addr(uint8_t* addr) { reg_write(DLA_IFMAP_ADDR_OFFSET, (uint32_t)(uintptr_t)addr); }

void set_filter_addr(int8_t* addr) { reg_write(DLA_FILTER_ADDR_OFFSET, (uint32_t)(uintptr_t)addr); }

void set_bias_addr(int32_t* addr) { reg_write(DLA_BIAS_ADDR_OFFSET, (uint32_t)(uintptr_t)addr); }

void set_opsum_addr(uint8_t* addr) { reg_write(DLA_OPSUM_ADDR_OFFSET, (uint32_t)(uintptr_t)addr); }

void set_glb_filter_addr(uint32_t addr) { reg_write(DLA_GLB_FILTER_ADDR_OFFSET, addr); }

void set_glb_bias_addr(uint32_t addr) { reg_write(DLA_GLB_BIAS_ADDR_OFFSET, addr); }

void set_glb_ofmap_addr(uint32_t addr) { reg_write(DLA_GLB_OFMAP_ADDR_OFFSET, addr); }

void set_input_activation_len(uint32_t len) { reg_write(DLA_IFMAP_LEN_OFFSET, len); };

void set_output_activation_len(uint32_t len) { reg_write(DLA_OFMAP_LEN_OFFSET, len); };

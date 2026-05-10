// driver_dla.cpp — DLA register-level driver.
//
// The DlaHAL instance is owned by tb.cpp. tb.cpp calls set_dla_hal(&hal) once
// after construction so that all register-level helpers below can reach the
// same HAL object.

#include "driver_dla.h"

#include <assert.h>
#include <stdio.h>

#include "dla_hal.hpp"

static DlaHAL* g_hal = nullptr;

void set_dla_hal(DlaHAL* hal) { g_hal = hal; }
DlaHAL* get_dla_hal() { return g_hal; }

void reg_write(uint32_t offset, uint32_t value) {
    assert(g_hal != nullptr);
    g_hal->memory_set(offset + DLA_MMIO_BASE_ADDR, value);
}

/* DLA configuration */
void set_enable(uint32_t scale_factor, bool maxpool, bool relu,
                bool operation) {
    uint32_t value;
    value = ((scale_factor&63) << 4) | (operation << 3) | (relu << 2) | (maxpool << 1) | (maxpool|relu);

    reg_write(DLA_ENABLE_OFFSET, value);
}

void set_mapping_param(uint32_t m, uint32_t e, uint32_t p, uint32_t q,
                       uint32_t r, uint32_t t) {
    uint32_t value;
    value = ((m&1023) << 16) | ((e&15) << 12) | ((p&7) << 9) | ((q&7) << 6) | ((r&7) << 3) | (t&7);


    reg_write(DLA_MAPPING_PARAM_OFFSET, value);
}

void set_shape_param1(uint32_t PAD, uint32_t U, uint32_t R, uint32_t S,
                      uint32_t C, uint32_t M) {
    uint32_t value;
    value = ((PAD&7) << 26) | ((U&3) << 24) | ((R&3) << 22) | ((S&3) << 20) | ((C&1023) << 10) | (M&1023);

    reg_write(DLA_SHAPE_PARAM1_OFFSET, value);
}

void set_shape_param2(uint32_t W, uint32_t H, uint32_t PAD) {
    uint32_t value;
    value = ((((PAD << 1) + W)&255) << 8) | (((PAD << 1) + H)&255);

    reg_write(DLA_SHAPE_PARAM2_OFFSET, value);
}

void set_ifmap_addr(uint8_t* addr) {
    reg_write(DLA_IFMAP_ADDR_OFFSET, (uint32_t)(uintptr_t)addr);
}

void set_filter_addr(int8_t* addr) {
    reg_write(DLA_FILTER_ADDR_OFFSET, (uint32_t)(uintptr_t)addr);
}

void set_bias_addr(int32_t* addr) {
    reg_write(DLA_BIAS_ADDR_OFFSET, (uint32_t)(uintptr_t)addr);
}

void set_opsum_addr(uint8_t* addr) {
    reg_write(DLA_OPSUM_ADDR_OFFSET, (uint32_t)(uintptr_t)addr);
}

void set_glb_filter_addr(uint32_t addr) {
    reg_write(DLA_GLB_FILTER_ADDR_OFFSET, addr);
}

void set_glb_bias_addr(uint32_t addr) {
    reg_write(DLA_GLB_BIAS_ADDR_OFFSET, addr);
}

void set_glb_ofmap_addr(uint32_t addr) {
    reg_write(DLA_GLB_OFMAP_ADDR_OFFSET, addr);
}

void set_input_activation_len(uint32_t len) {
    reg_write(DLA_IFMAP_LEN_OFFSET, len);
};

void set_output_activation_len(uint32_t len) {
    reg_write(DLA_OFMAP_LEN_OFFSET, len);
};

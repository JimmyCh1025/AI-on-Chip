// dla_hal.hpp — DLA (Eyeriss) Verilator simulation HAL.
//
// Encapsulates Vasic_wrapper, MMIO read/write, DMA handling, and
// interrupt-driven completion. Derives from HALBase.

#ifndef DLA_HAL_HPP
#define DLA_HAL_HPP

#include "Vasic_wrapper.h"
#include "hal.hpp"

// Advance the DLA DUT by one clock cycle and update counters
#ifdef USE_FST
#define clock_step(dut, signal, elapsed_cycle, elapsed_time) \
    do {                                                     \
        FST_FP->dump(elapsed_time);                          \
        (dut)->signal = 0;                                   \
        (dut)->eval();                                       \
        (elapsed_time) += CYCLE_TIME / 2;                    \
        FST_FP->dump(elapsed_time);                          \
        (dut)->signal = 1;                                   \
        (dut)->eval();                                       \
        (elapsed_time) += CYCLE_TIME / 2;                    \
        (elapsed_cycle)++;                                   \
    } while (0)
#else
#define clock_step(dut, signal, elapsed_cycle, elapsed_time) \
    do {                                                     \
        (dut)->signal = 0;                                   \
        (dut)->eval();                                       \
        (dut)->signal = 1;                                   \
        (dut)->eval();                                       \
        (elapsed_time) += CYCLE_TIME;                        \
        (elapsed_cycle)++;                                   \
    } while (0)
#endif

// AXI protocol constants used by DlaHAL's MMIO state machines
#define AXI_SIZE_BYTE 0b000
#define AXI_SIZE_HWORD 0b001
#define AXI_SIZE_WORD 0b010
#define AXI_BURST_INC 0x1
#define AXI_STRB_WORD 0b1111
#define AXI_STRB_HWORD 0b0011
#define AXI_STRB_BYTE 0b0001
#define AXI_RESP_OKAY 0x0
#define AXI_RESP_exOKAY 0x1
#define AXI_RESP_SLVERR 0x2
#define AXI_RESP_DECERR 0x3

#ifdef USE_FST
#include <verilated_fst_c.h>
#endif

constexpr uint32_t DLA_MAX_CYCLE = 100000;  // max simulation cycles
constexpr uint32_t DLA_RESET_CYCLE = 10;    // reset hold cycles
#define DLA_FST_FILE_NAME "ASIC.fst"
constexpr int DLA_TRACE_DEPTH = 3;

// DLA Hardware Abstraction Layer: lifecycle, MMIO, IRQ wait
class DlaHAL : public HALBase {
   private:
    struct runtime_info info_;
    uint32_t baseaddr_;
    uint32_t mmio_size_;
    Vasic_wrapper* device_;
    uint64_t vm_addr_h_;  // upper 32 bits of this object's address (for DMA)

    void handle_dma_read();
    void handle_dma_write();

#ifdef USE_FST
    VerilatedFstC* FST_FP = nullptr;
    int fst_task_id_ = 0;

    void fst_init();
    void fst_final();
#endif

   public:
    // @param baseaddr  MMIO base address.  @param mmio_size  MMIO region size.
    DlaHAL(uint32_t baseaddr, uint32_t mmio_size);
    ~DlaHAL() override;

    /* HALBase lifecycle */
    void init() override;
    void reset() override;
    void final() override;

    /* HALBase performance tracking */
    struct runtime_info get_runtime_info() const override;
    void reset_runtime_info() override;

    /* DLA-specific MMIO / IRQ */
    // Write a 32-bit value to an MMIO register; returns true on AXI OKAY.
    bool memory_set(uint32_t addr, uint32_t data);

    // Read a 32-bit value from an MMIO register; returns true on AXI OKAY.
    bool memory_get(uint32_t addr, uint32_t& data);

    // Block until DLA asserts its interrupt; service DMA requests while
    // waiting.
    void wait_for_irq();
};

#endif  // DLA_HAL_HPP

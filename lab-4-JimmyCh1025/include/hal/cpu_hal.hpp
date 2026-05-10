// cpu_hal.hpp — NutShell RV64IM CPU simulation driver.
//
// CpuHAL drives the Verilator DUT (clock edges, FST trace, D-cache snooping,
// ELF loading). AXI slave SMs and the run loop belong to tb.cpp.
// tb.cpp calls tick_negedge() / axi4_tick() / tick_posedge() each cycle.

#ifndef CPU_HAL_HPP
#define CPU_HAL_HPP

#include <verilated.h>
#include <verilated_fst_c.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "VNutShell.h"
#include "VNutShell___024root.h"
#include "elf_loader.h"
#include "hal.hpp"
#include "sram_model.h"

/* CPU-specific profiling metrics */
struct cpu_runtime_info : runtime_info {
    /* ---- L1 I-Cache ---- */
    uint64_t l1i_hit = 0;
    uint64_t l1i_miss = 0;
    /* ---- L1 D-Cache ---- */
    uint64_t l1d_hit = 0;
    uint64_t l1d_miss = 0;
    /* ---- L2 Cache ---- */
    uint64_t l2_hit = 0;
    uint64_t l2_miss = 0;
    /* ---- Miss penalty (total stall cycles while a miss is in-flight).
     * Divide by the miss count to get average stall cycles per miss. ---- */
    uint64_t l1d_miss_penalty_cycles = 0;
    uint64_t l2_miss_penalty_cycles = 0;
};

/* ---------------------------------------------------------------
 * CpuHAL
 * --------------------------------------------------------------- */
class CpuHAL : public HALBase {
   public:
    // @param sram  Sparse SRAM model owned by tb.cpp.
    explicit CpuHAL(SRAMModel& sram);
    ~CpuHAL() override;

    // Load ELF, optionally enable FST trace, then assert reset.
    void init(const char* elf_path, bool enable_trace = false,
              const char* trace_path = "cpu_trace.fst");

    /* HALBase lifecycle overrides */
    void init() override;   // no-arg (unused by CPU path)
    void reset() override;  // reset with default 20 cycles
    void final() override;  // close FST + free DUT

    // Drive reset for `cycles` clock cycles, then de-assert.
    void reset(int cycles);

    // Expose raw DUT for tb.cpp to drive io_mem_* pins directly.
    VNutShell* dut() { return dut_; }

    // clock=0, eval, FST dump. Call BEFORE driving AXI pins.
    void tick_negedge();
    // clock=1, eval, FST dump, advance counter. Call AFTER driving AXI pins.
    void tick_posedge();

    // Sync D-Cache to SRAM, then copy `len` bytes at `addr` into `buf`.
    void read_output(uint32_t addr, uint8_t* buf, uint32_t len);

    // Look up a symbol's virtual address from the loaded ELF (0 if not found).
    uint32_t get_symbol_addr(const char* name);

    // CPU-specific runtime info with cache stats
    struct runtime_info get_runtime_info() const override;
    struct cpu_runtime_info get_cpu_runtime_info() const;
    void reset_runtime_info() override;

    // Returns true when done_flag 0xDEAD is written to 0x80000004.
    bool check_done();
    // Returns true when mepc is stuck for several consecutive cycles.
    bool check_halted();
    // Stop FST recording.
    void stop_trace();
    // Accumulate AXI DRAM byte counts (called by tb.cpp after each AXI xact).
    // Cache hit/miss counters are updated automatically inside tick_posedge().
    void update_memory_stats(uint32_t rd, uint32_t wr);

   private:
    VNutShell* dut_ = nullptr;
    VerilatedFstC* tfp_ = nullptr;
    struct cpu_runtime_info info_ = {};

    SRAMModel& sram_;

    /* halt detection */
    uint64_t halt_prev_mepc_ = 0;
    int halt_count_ = 0;

    int64_t l1d_miss_start_cycle_ =
        -1;  ///< Cycle when the last L1D miss started (-1 = idle).
    int64_t l2_miss_start_cycle_ =
        -1;  ///< Cycle when the last L2  miss started (-1 = idle).

    /** Read addr from D-Cache first, fall back to SRAM model. */
    uint64_t peek_dcache(uint32_t addr);

    /** Flush all dirty D-cache lines back to SRAM model. */
    void sync_dcache_to_sram();
};

#endif  // CPU_HAL_HPP

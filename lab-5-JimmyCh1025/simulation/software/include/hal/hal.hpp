/**
 * @file hal.hpp
 * @brief Defines simulation-related macros for a Verilog DUT testbench.
 *
 * These macros control clock stepping, timing parameters, and VCD (waveform)
 * dumping for Verilog simulation.
 */

#ifndef HAL_HPP
#define HAL_HPP

#include <stdint.h>
#include <stdlib.h>

#include <map>
#include <vector>

#include "Vasic_wrapper.h"
#include "verilated_vcd_c.h"

/// @brief Defines the duration of one clock cycle in nanoseconds.
#define CYCLE_TIME 5  // ns

/// @brief Maximum simulation cycles before timeout.
#define MAX_CYCLE 100000

/// @brief Number of cycles to hold the reset signal active.
#define RESET_CYCLE 10

/// @brief Memory access latency in clock cycles.
#define MEM_ACCESS_CYCLE 5

/// @brief VCD (Value Change Dump) file name for waveform dumping.
#define VCD_FILE_NAME "ASIC.vcd"

/// @brief Depth of the waveform trace hierarchy.
#define TRACE_DEPTH 3

#ifndef USE_VCD
/**
 * @brief Advances the clock cycle for a Verilog DUT (Device Under Test).
 *
 * This macro toggles the specified signal in the DUT to simulate a clock cycle.
 * It increments both the elapsed time and cycle count accordingly.
 *
 * @param  dut            Pointer to the DUT instance.
 * @param  signal         The clock signal to toggle.
 * @param  elapsed_cycle  Variable tracking the number of elapsed cycles.
 * @param  elapsed_time   Variable tracking the elapsed simulation time.
 *
 * @note  When `USE_VCD` is not defined, the macro executes a full clock cycle
 *        in a single step (falling + rising edge).
 *
 * @warning Ensure `CYCLE_TIME` is defined before using this macro.
 */
#define clock_step(dut, signal, elapsed_cycle, elapsed_time) \
  dut->signal = 0;                                           \
  dut->eval(); /* falling edge */                            \
  dut->signal = 1;                                           \
  dut->eval(); /* rising edge */                             \
  (elapsed_time) += CYCLE_TIME;                              \
  (elapsed_cycle)++;

#else  // USE_VCD is defined

/**
 * @brief Advances the clock cycle and dumps waveform data when `USE_VCD` is
 * enabled.
 *
 * This macro toggles the specified signal in the DUT, but also records the
 * simulation time to a VCD (waveform) file at each transition.
 *
 * @param  dut            Pointer to the DUT instance.
 * @param  signal         The clock signal to toggle.
 * @param  elapsed_cycle  Variable tracking the number of elapsed cycles.
 * @param  elapsed_time   Variable tracking the elapsed simulation time.
 *
 * @note  When `USE_VCD` is defined, the macro splits the clock cycle into two
 *        half-cycles and dumps waveforms at each transition.
 *
 * @warning Ensure that `VCD_FP` is properly initialized before using this
 * macro.
 */
#define clock_step(dut, signal, elapsed_cycle, elapsed_time) \
  VCD_FP->dump(elapsed_time);                                \
  dut->signal = 0;                                           \
  dut->eval(); /* falling edge */                            \
  (elapsed_time) += CYCLE_TIME / 2;                          \
  VCD_FP->dump(elapsed_time);                                \
  dut->signal = 1;                                           \
  dut->eval(); /* rising edge */                             \
  (elapsed_time) += CYCLE_TIME / 2;                          \
  (elapsed_cycle)++;

#endif  // USE_VCD

/**
 * @struct runtime_info
 * @brief Stores runtime performance metrics for a hardware simulation or
 * execution.
 *
 * This structure holds information about execution cycles, elapsed time, and
 * memory operations during a hardware simulation or a specific computation.
 */
struct runtime_info {
  uint32_t elapsed_cycle;  ///< Number of cycles elapsed during execution.
  uint32_t elapsed_time;   ///< Total elapsed time (e.g., in nanoseconds).
  uint32_t memory_read;    ///< Total number of memory read operations.
  uint32_t memory_write;   ///< Total number of memory write operations.
};

/**
 * @class HardwareAbstractionLayer
 * @brief Provides an abstraction layer for interacting with a hardware device.
 *
 * This class manages memory-mapped I/O (MMIO), Direct Memory Access (DMA),
 * and runtime performance tracking for a hardware accelerator.
 * It also supports waveform (VCD) dumping if enabled.
 */
class HardwareAbstractionLayer {
  private:
  struct MemoryRegion {
    uint32_t dev_addr_start;
    uint32_t dev_addr_end;
    uint8_t* host_ptr;
  };

  std::vector<MemoryRegion> memory_regions;
  uint32_t                  next_alloc_addr = 0x10000000;

  ///< Stores execution cycle, time, and memory access stats.
  struct runtime_info info;

  uint32_t baseaddr;   ///< Base address of the memory-mapped I/O (MMIO) region
  uint32_t mmio_size;  ///< Size of the MMIO region.

  Vasic_wrapper* device;     ///< Pointer to the hardware device wrapper.
  uint64_t       vm_addr_h;  ///< Virtual memory address for hardware interaction.

  /**
   * @brief Handles DMA read operations.
   * @note This is an internal function and should not be called directly.
   */
  void handle_dma_read();

  /**
   * @brief Handles DMA write operations.
   * @note This is an internal function and should not be called directly.
   */
  void handle_dma_write();

#ifdef USE_VCD
  VerilatedVcdC* VCD_FP;       ///< Pointer to the VCD file handler.
  int            vcd_task_id;  ///< Task ID associated with waveform dumping.

  /**
   * @brief Initializes VCD (waveform) dumping.
   */
  void vcd_init();

  /**
   * @brief Finalizes VCD (waveform) dumping.
   */
  void vcd_final();
#endif

  public:
  /**
   * @brief Constructs the HAL with the specified base address and MMIO size.
   * @param baseaddr The base address of the hardware device.
   * @param mmio_size The size of the MMIO region.
   */
  HardwareAbstractionLayer(uint32_t baseaddr, uint32_t mmio_size);

  /**
   * @brief Destructor for the HAL, ensuring proper cleanup.
   */
  ~HardwareAbstractionLayer();

  /**
   * @brief Provide API for Runtime
   * @param host_ptr Real data address
   * @param size Data size
   * @return Handle for hardware (uint32_t)
   */
  uint32_t mmu_map(void* host_ptr, size_t size);

  /**
   * @brief Release mapping
   */
  void mmu_unmap(uint32_t device_addr);

  /**
   * @brief Writes data to a specific MMIO address.
   * @param addr The address to write to.
   * @param data The data to be written.
   * @return True if the write operation is successful, false otherwise.
   */
  bool memory_set(uint32_t addr, uint32_t data);

  /**
   * @brief Reads data from a specific MMIO address.
   * @param addr The address to read from.
   * @param data Reference to store the read data.
   * @return True if the read operation is successful, false otherwise.
   */
  bool memory_get(uint32_t addr, uint32_t& data);

  /**
   * @brief Waits for an interrupt request (IRQ) from the hardware.
   * @note This function blocks execution until an interrupt occurs.
   */
  void wait_for_irq();

  /**
   * @brief Retrieves the current runtime performance information.
   * @return A `runtime_info` structure containing performance metrics.
   */
  struct runtime_info get_runtime_info();

  /**
   * @brief Resets runtime performance counters to zero.
   */
  void reset_runtime_info();

  /**
   * @brief Initializes the hardware and prepares it for operation.
   */
  void init();

  /**
   * @brief Shuts down the hardware and releases resources.
   */
  void final();
};

#endif  // HAL_HPP

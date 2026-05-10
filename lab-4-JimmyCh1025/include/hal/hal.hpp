// hal.hpp — shared types and HALBase interface.

#ifndef HAL_HPP
#define HAL_HPP

#include <cstdint>

/* Simulation timing constants */
constexpr uint32_t CYCLE_TIME = 5;        // ns per cycle
constexpr uint32_t MEM_ACCESS_CYCLE = 5;  // memory-access latency in cycles

/* Performance metrics common to all HAL implementations. */
struct runtime_info {
    uint64_t elapsed_cycle;  ///< Total clock cycles elapsed
    uint64_t elapsed_time;   ///< Total elapsed time in nanoseconds
    uint32_t memory_read;    ///< Bytes transferred in  (DLA: DMA reads, CPU: L2
                             ///< miss refills)
    uint32_t memory_write;  ///< Bytes transferred out (DLA: DMA writes, CPU: L2
                            ///< dirty evictions)
};

/* Abstract base class for DlaHAL and CpuHAL. */
class HALBase {
   public:
    virtual ~HALBase() = default;

    /* ---- Lifecycle ---- */
    virtual void init() = 0;
    virtual void reset() = 0;
    virtual void final() = 0;

    /* ---- Performance tracking ---- */
    virtual struct runtime_info get_runtime_info() const = 0;
    virtual void reset_runtime_info() = 0;
};

#endif  // HAL_HPP

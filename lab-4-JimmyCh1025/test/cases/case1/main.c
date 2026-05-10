#include <stdint.h>

#include "runtime.h"
#include "workload.h"

/*
 * This file is compiled into a bare-metal RV64 ELF that runs on the
 * NutShell CPU Verilator model driven by `test/testbench/cpu/tb.cpp`.
 *
 *   1. Call run_workload() (defined in workload.c) to execute the
 *      backend-agnostic computation using the CPU runtime API.
 *   2. Write 0xDEAD to 0x80000004 to signal the testbench that
 *      execution has finished.
 *
 * Golden verification is handled entirely by the host-side tb.cpp;
 * this ELF only produces the output[] result in SRAM.
 */

int main(void) {
    run_workload();
    /* Signal CPU testbench: computation done */
    *((volatile uint32_t*)0x80000004) = 0xDEAD;
    while (1) {
    }
    return 0;
}

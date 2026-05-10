/* tb.cpp — DLA Testbench driver
 *
 * Case 0-1: data.h provides golden output[] arrays; after run_workload(),
 * compare output[] vs golden array element-wise.
 *
 * Case 2:   data.h provides pre-quantized tensors loaded by workload.c;
 *           after run_workload(), dequantize output and compare vs host
 *           conv_maxpooling reference within a 2-LSB float tolerance.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dla_hal.hpp"
#include "driver_dla.h"
#include "hal.hpp"
#include "runtime.h"

#define COL_RESET "\033[0m"
#define COL_GREY "\033[0;37m"
#define COL_WHITE "\033[0;37m"
#define COL_GREEN "\033[0;32m"
#define COL_RED "\033[0;31m"

#define LOG_INFO(fmt, ...) \
    fprintf(stdout, COL_GREY fmt COL_RESET "\n", ##__VA_ARGS__)
#define LOG_OK(fmt, ...) \
    fprintf(stdout, COL_GREEN fmt COL_RESET "\n", ##__VA_ARGS__)
#define LOG_ERR(fmt, ...) \
    fprintf(stderr, COL_RED fmt COL_RESET "\n", ##__VA_ARGS__)
#define LOG_RES(fmt, ...) \
    fprintf(stdout, COL_WHITE fmt COL_RESET "\n", ##__VA_ARGS__)

/* Per-case workload dispatch — one workload.c linked per binary. */
#if CASE_NUM == 2
#include "kernel_cpu.h"
#include "quantize.h"

/* case2 tensor dimensions (matches workload.c / data.h) */
#define CASE2_IN_N 16384
#define CASE2_WT_N 110592
#define CASE2_BIAS_N 192
#define CASE2_OUT_N 12288
#define CASE2_SCALE 7
#define CASE2_FLOAT_TOL (2.0f / 128.0f) /* ≤2 LSBs at scale=7 */
#endif                                  /* CASE_NUM == 2 */

#if CASE_NUM == 0
#include "../../../test/cases/case0/workload.h"
static int check_case(void) {
    int err = 0;
    for (int i = 0; i < 1024; i++) {
        if (output[i] != conv_relu_ans_golden[i]) {
            LOG_ERR("[ERR case0] idx=%d, out=%u, expected=%u, abs_diff=%d", i,
                    output[i], conv_relu_ans_golden[i],
                    abs((int)output[i] - (int)conv_relu_ans_golden[i]));
            err++;
        }
    }
    return err;
}

#elif CASE_NUM == 1
#include "../../../test/cases/case1/workload.h"
static int check_case(void) {
    int err = 0;
    for (int i = 0; i < 16384; i++) {
        if (output[i] != conv_relu_max_ans_golden[i]) {
            LOG_ERR("[ERR case1] idx=%d, out=%u, expected=%u, abs_diff=%d", i,
                    output[i], conv_relu_max_ans_golden[i],
                    abs((int)output[i] - (int)conv_relu_max_ans_golden[i]));
            err++;
        }
    }
    return err;
}

#elif CASE_NUM == 2
#include "../../../test/cases/case2/workload.h"
static int check_case(void) { return 0; }

#else
#error "CASE_NUM must be defined to 0, 1, or 2"
#endif

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    char label[64];
    snprintf(label, sizeof(label), "case%d DLA", CASE_NUM);

    LOG_INFO("[TB/DLA] Starting %s simulation...", label);

    /* Initialise DLA — dla tb owns the DlaHAL object, just like cpu tb owns
     * CpuHAL. Static storage ensures vm_addr_h_ (upper 32b of this) matches the
     * upper 32b of global data arrays used for DMA. */
    static DlaHAL hal(DLA_MMIO_BASE_ADDR, DLA_MMIO_SIZE);
    set_dla_hal(&hal);  // inject into register-level driver (driver_dla)
    hal.init();

    /* Run the workload */
    run_workload();

    struct runtime_info ri = hal.get_runtime_info();
    hal.final();

    int err = 0;

#if CASE_NUM == 2
    /* Rebuild host reference: regenerate the same float data that produced
     * data.h, re-quantize, run conv_maxpooling on the host CPU, dequantize
     * both sides and compare element-wise within CASE2_FLOAT_TOL. */
    {
        const float fp = (float)(1 << CASE2_SCALE);
        uint8_t* q_act = (uint8_t*)malloc(CASE2_IN_N);
        int8_t* q_wt = (int8_t*)malloc(CASE2_WT_N);
        int32_t* q_bias = (int32_t*)malloc(CASE2_BIAS_N * sizeof(int32_t));

        {
            float* f_act = (float*)malloc(CASE2_IN_N * sizeof(float));
            for (int i = 0; i < CASE2_IN_N; i++)
                f_act[i] = (float)((i % 32) + 112 - 128) / fp;
            quantize(f_act, q_act, CASE2_IN_N, CASE2_SCALE);
            free(f_act);

            float* f_wt = (float*)malloc(CASE2_WT_N * sizeof(float));
            for (int i = 0; i < CASE2_WT_N; i++)
                f_wt[i] = (float)((i % 11) - 5) / fp;
            quantize_weights(f_wt, q_wt, CASE2_WT_N, CASE2_SCALE);
            free(f_wt);

            for (int i = 0; i < CASE2_BIAS_N; i++)
                q_bias[i] = (int32_t)(10 * (i % 32) + 128);
        }

        /* Reference golden via host CPU implementation */
        uint8_t* ref_out = (uint8_t*)malloc(CASE2_OUT_N);
        conv_maxpooling(64, 16, 16, q_act, 192, 64, 3, 3, q_wt, q_bias, 1,
                        ref_out, CASE2_SCALE);

        /* Dequantize and compare */
        float* f_got = (float*)malloc(CASE2_OUT_N * sizeof(float));
        float* f_ref = (float*)malloc(CASE2_OUT_N * sizeof(float));
        dequantize(output, f_got, CASE2_OUT_N, CASE2_SCALE);
        dequantize(ref_out, f_ref, CASE2_OUT_N, CASE2_SCALE);

        for (int i = 0; i < CASE2_OUT_N; i++) {
            float diff = (float)fabs((double)(f_got[i] - f_ref[i]));
            if (diff > CASE2_FLOAT_TOL) {
                LOG_ERR(
                    "[ERR case2] idx=%d, out=%.4f, expected=%.4f, diff=%.4f"
                    " (raw out=%u, expected=%u)",
                    i, f_got[i], f_ref[i], diff, output[i], ref_out[i]);
                err++;
            }
        }

        free(q_act);
        free(q_wt);
        free(q_bias);
        free(ref_out);
        free(f_got);
        free(f_ref);
    }
#else
    err = check_case();
#endif

    printf("\n");
    LOG_RES("===== DLA Simulation Result =====");
    LOG_RES("  Case              : %s", label);
    LOG_RES("  Cycles            : %llu", (unsigned long long)ri.elapsed_cycle);
    LOG_RES("  Time (s)          : %f", (float)ri.elapsed_time / 1000000000);
    LOG_RES("  Mem reads (Bytes) : %u", ri.memory_read);
    LOG_RES("  Mem writes (Bytes): %u", ri.memory_write);
    LOG_RES("  Errors            : %d  %s", err,
            err == 0 ? "[PASS]" : "[FAIL]");
    LOG_RES("=================================");

    if (err == 0) {
        LOG_OK("[TB/DLA] *** TEST PASSED  (cycles=%llu) ***",
               (unsigned long long)ri.elapsed_cycle);
    } else {
        LOG_ERR("[TB/DLA] *** FAILED  err=%d ***", err);
    }
    printf("\n");

    return err == 0 ? 0 : 1;
}

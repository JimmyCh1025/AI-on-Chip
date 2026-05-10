/* tb.cpp — CPU Testbench driver
 * Case 0-2: load ELF → run until done_flag (0xDEAD) → compare output[] vs
 *            golden array in data.h.
 *
 * Case 3:   data.h provides pre-quantized tensors; after run_workload(),
 * rebuild host reference and compare dequantized output within 2-LSB tolerance.
 *
 * Case 4-5: case_cpu_fallback (linear / linear+relu), golden in data.h.
 */

#include <verilated.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "cpu_hal.hpp"
#include "kernel_cpu.h"
#include "quantize.h"

#define COL_RESET "\033[0m"
#define COL_GREY "\033[0;37m"
#define COL_GREEN "\033[0;32m"
#define COL_RED "\033[0;31m"
#define COL_CYAN "\033[0;36m"

#define LOG_INFO(fmt, ...) \
    fprintf(stdout, COL_GREY fmt COL_RESET "\n", ##__VA_ARGS__)
#define LOG_OK(fmt, ...) \
    fprintf(stdout, COL_GREEN fmt COL_RESET "\n", ##__VA_ARGS__)
#define LOG_ERR(fmt, ...) \
    fprintf(stderr, COL_RED fmt COL_RESET "\n", ##__VA_ARGS__)
#define LOG_RES(fmt, ...) \
    fprintf(stdout, COL_GREY fmt COL_RESET "\n", ##__VA_ARGS__)
#define LOG_DBG(fmt, ...) \
    fprintf(stdout, COL_CYAN fmt COL_RESET "\n", ##__VA_ARGS__)

namespace case0 {
#include "../../../test/cases/case0/data.h"
}

#undef DATA_H
namespace case1 {
#include "../../../test/cases/case1/data.h"
}

#undef DATA_H
namespace case_cpu_fallback {
#include "../../../test/cases/case_cpu_fallback/data.h"
}

static constexpr int kDiagPrintLimit =
    16; /* max mismatches to print when DIAG=1 */

static void decode_chw(int idx, int out_h, int out_w, int* c, int* h, int* w) {
    int hw = out_h * out_w;
    *c = idx / hw;
    int rem = idx % hw;
    *h = rem / out_w;
    *w = rem % out_w;
}

static int compare_u8_tensor(const char* case_name, const uint8_t* got,
                             const uint8_t* expected, int len, int out_h,
                             int out_w, bool enable_diag) {
    int err = 0, printed = 0;
    int first_idx = -1, max_idx = -1;
    int max_abs_diff = -1;
    uint8_t first_got_v = 0, first_exp_v = 0;

    for (int i = 0; i < len; i++) {
        if (got[i] == expected[i]) continue;

        err++;
        int abs_diff = abs((int)got[i] - (int)expected[i]);
        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
            max_idx = i;
        }
        if (first_idx < 0) {
            first_idx = i;
            first_got_v = got[i];
            first_exp_v = expected[i];
        }

        if (enable_diag && printed < kDiagPrintLimit) {
            int c, h, w;
            decode_chw(i, out_h, out_w, &c, &h, &w);
            LOG_ERR(
                "[ERR %s] idx=%d (c=%d, h=%d, w=%d) got=%u, expected=%u, "
                "abs_diff=%d",
                case_name, i, c, h, w, got[i], expected[i], abs_diff);
            printed++;
        }
    }

    if (err > 0) {
        int c0, h0, w0, c1, h1, w1;
        decode_chw(first_idx, out_h, out_w, &c0, &h0, &w0);
        decode_chw(max_idx, out_h, out_w, &c1, &h1, &w1);
        LOG_ERR(
            "\n[DIAG %s] mismatches=%d:\n"
            "\tfirst at (idx=%d, c=%d, h=%d, w=%d, got=%u, expected=%u)\n"
            "\tmax_abs_diff=%d at (idx=%d, c=%d, h=%d, w=%d)",
            case_name, err, first_idx, c0, h0, w0, first_got_v, first_exp_v,
            max_abs_diff, max_idx, c1, h1, w1);
        if (!enable_diag)
            LOG_DBG("[TB/CPU] Hint: add DIAG=1 to print first %d mismatches.",
                    kDiagPrintLimit);
    }

    return err;
}

static int compare_f32_tensor(const char* case_name, const float* got,
                              const float* expected, const uint8_t* raw_got,
                              const uint8_t* raw_expected, int len, int out_h,
                              int out_w, float tol, bool enable_diag) {
    int err = 0, printed = 0;
    int first_idx = -1, max_idx = -1;
    float first_got_v = 0.0f, first_exp_v = 0.0f, max_diff = -1.0f;

    for (int i = 0; i < len; i++) {
        float diff = fabsf(got[i] - expected[i]);
        if (diff <= tol) continue;

        err++;
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
        if (first_idx < 0) {
            first_idx = i;
            first_got_v = got[i];
            first_exp_v = expected[i];
        }

        if (enable_diag && printed < kDiagPrintLimit) {
            int c, h, w;
            decode_chw(i, out_h, out_w, &c, &h, &w);
            LOG_ERR(
                "[ERR %s] idx=%d (c=%d, h=%d, w=%d) got=%.4f, expected=%.4f, "
                "diff=%.4f (raw got=%u, expected=%u)",
                case_name, i, c, h, w, got[i], expected[i], diff, raw_got[i],
                raw_expected[i]);
            printed++;
        }
    }

    if (err > 0) {
        int c0, h0, w0, c1, h1, w1;
        decode_chw(first_idx, out_h, out_w, &c0, &h0, &w0);
        decode_chw(max_idx, out_h, out_w, &c1, &h1, &w1);
        LOG_ERR(
            "\n[DIAG %s] mismatches=%d:\n"
            "\tfirst at (idx=%d, c=%d, h=%d, w=%d, got=%.4f, expected=%.4f)\n"
            "\tmax_abs_diff=%.4f at (idx=%d, c=%d, h=%d, w=%d)",
            case_name, err, first_idx, c0, h0, w0, first_got_v, first_exp_v,
            max_diff, max_idx, c1, h1, w1);
        if (!enable_diag)
            LOG_DBG("[TB/CPU] Hint: add DIAG=1 to print first %d mismatches.",
                    kDiagPrintLimit);
    }

    return err;
}

/* case2 tensor dimensions (matches workload.c / data.h) */
static constexpr int CASE2_IN_N = 16384;
static constexpr int CASE2_WT_N = 110592;
static constexpr int CASE2_BIAS_N = 192;
static constexpr int CASE2_OUT_N = 12288;
static constexpr int CASE2_SCALE = 7;
static constexpr float CASE2_FLOAT_TOL = 2.0f / 128.0f; /* ≤2 LSBs at scale=7 */

/* Build quantized reference tensors and run host conv+maxpool for case2. */
static void build_case2_reference(std::vector<uint8_t>& ref_out) {
    const float fp = ldexpf(1.0f, CASE2_SCALE);

    std::vector<float> f_act(CASE2_IN_N);
    std::vector<uint8_t> q_act(CASE2_IN_N);
    for (int i = 0; i < CASE2_IN_N; i++)
        f_act[i] = (float)((i % 32) + 112 - 128) / fp;
    quantize(f_act.data(), q_act.data(), CASE2_IN_N, CASE2_SCALE);

    std::vector<float> f_wt(CASE2_WT_N);
    std::vector<int8_t> q_wt(CASE2_WT_N);
    for (int i = 0; i < CASE2_WT_N; i++) f_wt[i] = (float)((i % 11) - 5) / fp;
    quantize_weights(f_wt.data(), q_wt.data(), CASE2_WT_N, CASE2_SCALE);

    std::vector<int32_t> q_bias(CASE2_BIAS_N);
    for (int i = 0; i < CASE2_BIAS_N; i++)
        q_bias[i] = (int32_t)(10 * (i % 32) + 128);

    ref_out.resize(CASE2_OUT_N);
    conv_maxpooling(64, 16, 16, q_act.data(), 192, 64, 3, 3, q_wt.data(),
                    q_bias.data(), 1, ref_out.data(), CASE2_SCALE);
}

static void axi4_tick(CpuHAL& cpu, SRAMModel& sram, AXI4ReadSM& rd_sm,
                      AXI4WriteSM& wr_sm) {
    VNutShell* dut = cpu.dut();
    rd_sm.tick_delay();
    wr_sm.tick_delay();

    /* --- AXI4 read channel --- */
    dut->io_mem_ar_ready = rd_sm.arready() ? 1 : 0;
    if (dut->io_mem_ar_valid && rd_sm.arready()) {
        rd_sm.accept(dut->io_mem_ar_bits_addr, dut->io_mem_ar_bits_len);
        cpu.update_memory_stats((dut->io_mem_ar_bits_len + 1) * 8, 0);
    }
    if (rd_sm.valid()) {
        dut->io_mem_r_valid = 1;
        dut->io_mem_r_bits_data = sram.read(rd_sm.data_addr());
        dut->io_mem_r_bits_last = rd_sm.is_last() ? 1 : 0;
        rd_sm.advance();
    } else {
        dut->io_mem_r_valid = 0;
        dut->io_mem_r_bits_data = 0;
        dut->io_mem_r_bits_last = 0;
    }

    /* --- AXI4 write channel --- */
    dut->io_mem_aw_ready = wr_sm.awready() ? 1 : 0;
    if (dut->io_mem_aw_valid && wr_sm.awready()) {
        wr_sm.accept_addr(dut->io_mem_aw_bits_addr, dut->io_mem_aw_bits_len);
        cpu.update_memory_stats(0, (dut->io_mem_aw_bits_len + 1) * 8);
    }
    dut->io_mem_w_ready = wr_sm.wready() ? 1 : 0;
    if (dut->io_mem_w_valid && wr_sm.wready())
        wr_sm.accept_data(dut->io_mem_w_bits_data, 0xFF,
                          dut->io_mem_w_bits_last, sram);
    dut->io_mem_b_valid = wr_sm.bvalid() ? 1 : 0;
    if (wr_sm.bvalid()) wr_sm.accept_resp();

    /* --- Unused / tie-off signals --- */
    dut->io_mmio_req_ready = 1;
    dut->io_mmio_resp_valid = 0;
    dut->io_mmio_resp_bits_cmd = 0;
    dut->io_mmio_resp_bits_rdata = 0;

    dut->io_frontend_aw_ready = 0;
    dut->io_frontend_w_ready = 0;
    dut->io_frontend_b_valid = 0;
    dut->io_frontend_ar_ready = 0;
    dut->io_frontend_r_valid = 0;
    dut->io_frontend_r_bits_data = 0;

    dut->io_meip = 0;
}

static bool run_simulation(CpuHAL& cpu, SRAMModel& sram, AXI4ReadSM& rd_sm,
                           AXI4WriteSM& wr_sm, uint64_t max_cycles,
                           uint64_t fst_stop_cycle) {
    LOG_INFO("[TB/CPU] Running (max %llu cycles)...",
             (unsigned long long)max_cycles);

    for (uint64_t c = 0; c < max_cycles; c++) {
        cpu.tick_negedge();

        uint64_t cycle = cpu.get_runtime_info().elapsed_cycle;
        axi4_tick(cpu, sram, rd_sm, wr_sm);

        cpu.tick_posedge();

        if (fst_stop_cycle > 0 && cycle >= fst_stop_cycle) cpu.stop_trace();

        if (cpu.check_done()) {
            LOG_INFO("[TB/CPU] Done flag at cycle %llu",
                     (unsigned long long)cpu.get_runtime_info().elapsed_cycle);
            return true;
        }

        if (cpu.check_halted()) {
            uint64_t cur = cpu.get_runtime_info().elapsed_cycle;
            LOG_INFO("[TB/CPU] Halt detected at cycle %llu",
                     (unsigned long long)cur);
            if (cpu.check_done()) return true;
            LOG_ERR("[TB/CPU] WARNING: halted without done_flag");
            return false;
        }
    }

    LOG_ERR("[TB/CPU] TIMEOUT after %llu cycles!",
            (unsigned long long)max_cycles);
    return false;
}

struct CaseInfo {
    const char* name;
    uint32_t output_len;
    int output_h;
    int output_w;
    const uint8_t* golden;
};

static const CaseInfo CASE_TABLE[5] = {
    {"case0", 1024, 8, 8, case0::conv_relu_ans_golden},
    {"case1", 16384, 16, 16, case1::conv_relu_max_ans_golden},
    {"case2", CASE2_OUT_N, 8, 8, nullptr},
    {"case_cpu_fallback_linear", 256, 1, 1, case_cpu_fallback::linear_golden},
    {"case_cpu_fallback_linear_relu", 256, 1, 1,
     case_cpu_fallback::linear_relu_golden},
};

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::randReset(0);

    const char* elf_file = nullptr;
    bool enable_trace = false;
    bool enable_diag = false;
    const char* trace_file = "cpu_trace.fst";
    int case_num = -1;
    bool use_improve = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--trace") == 0)
            enable_trace = true;
        else if (strcmp(argv[i], "--trace-file") == 0 && i + 1 < argc)
            trace_file = argv[++i];
        else if (strcmp(argv[i], "--diag") == 0)
            enable_diag = true;
        else if (strcmp(argv[i], "--improve") == 0)
            use_improve = true;
        else if (strcmp(argv[i], "--elf") == 0 && i + 1 < argc)
            elf_file = argv[++i];
        else if (strcmp(argv[i], "--case") == 0 && i + 1 < argc)
            case_num = atoi(argv[++i]);
        else {
            fprintf(stderr,
                    "[TB/CPU] Unknown argument: %s\n"
                    "Usage: %s --case <N> [--improve]\n"
                    "       %s --elf <file.elf>\n"
                    "       [--trace] [--diag]\n",
                    argv[i], argv[0], argv[0]);
            return 1;
        }
    }

    char case_elf_buf[512];
    if (elf_file == nullptr) {
        if (case_num < 0) {
            fprintf(stderr,
                    "Usage: %s --case <N> [--improve]\n"
                    "       %s --elf <file.elf>\n"
                    "       [--trace] [--diag]\n",
                    argv[0], argv[0]);
            return 1;
        }
        const char* variant = use_improve ? "improve" : "original";
        if (case_num <= 2)
            snprintf(case_elf_buf, sizeof(case_elf_buf),
                     "../../../src/runtime/cpu/case%d_%s.elf", case_num,
                     variant);
        else
            snprintf(case_elf_buf, sizeof(case_elf_buf),
                     "../../../src/runtime/cpu/%s_%s.elf",
                     CASE_TABLE[case_num].name, variant);
        elf_file = case_elf_buf;
    }

    char label[512];
    if (case_num >= 0) {
        const char* variant = use_improve ? "improve" : "original";
        if (case_num <= 2)
            snprintf(label, sizeof(label), "case%d %s", case_num, variant);
        else
            snprintf(label, sizeof(label), "%s %s", CASE_TABLE[case_num].name,
                     variant);
    } else {
        snprintf(label, sizeof(label), "%s", elf_file);
    }

    SRAMModel sram;
    AXI4ReadSM rd_sm;
    AXI4WriteSM wr_sm;
    CpuHAL cpu(sram);

    cpu.init(elf_file, enable_trace, trace_file);
    cpu.reset(20);

    bool done = run_simulation(cpu, sram, rd_sm, wr_sm, 2000000000UL, 0);

    if (enable_trace)
        LOG_DBG("[TB/CPU] FST waveform dumped to: %s", trace_file);

    if (!done) {
        LOG_ERR("[TB/CPU] Program did not complete within timeout.");
    }

    struct cpu_runtime_info ri = cpu.get_cpu_runtime_info();

    int err = 0;
    if (case_num >= 0 && case_num <= 5) {
        const CaseInfo& ci = CASE_TABLE[case_num];

        uint32_t output_addr = cpu.get_symbol_addr("output");
        if (output_addr == 0) {
            LOG_ERR("[TB/CPU] ERROR: could not resolve 'output' symbol in ELF");
            return 1;
        }

        std::vector<uint8_t> out_buf(ci.output_len);
        cpu.read_output(output_addr, out_buf.data(), ci.output_len);

        if (case_num == 2) {
            std::vector<uint8_t> ref_out;
            build_case2_reference(ref_out);

            std::vector<float> f_dut(CASE2_OUT_N), f_ref(CASE2_OUT_N);
            dequantize(out_buf.data(), f_dut.data(), CASE2_OUT_N, CASE2_SCALE);
            dequantize(ref_out.data(), f_ref.data(), CASE2_OUT_N, CASE2_SCALE);

            err =
                compare_f32_tensor("case2", f_dut.data(), f_ref.data(),
                                   out_buf.data(), ref_out.data(), CASE2_OUT_N,
                                   8, 8, CASE2_FLOAT_TOL, enable_diag);
        } else {
            err = compare_u8_tensor(ci.name, out_buf.data(), ci.golden,
                                    (int)ci.output_len, ci.output_h,
                                    ci.output_w, enable_diag);
        }
    }

    printf("\n");
    LOG_RES("===== CPU Simulation Result =====");
    LOG_RES("  Case          : %s", label);
    LOG_RES("  Cycles        : %llu", (unsigned long long)ri.elapsed_cycle);
    LOG_RES("  Time (s)      : %.6f", (float)ri.elapsed_time / 1000000000);

    /* ---- Cache statistics ---- */
    auto log_cache_hit = [&](const char* name, uint64_t hit, uint64_t miss) {
        uint64_t total = hit + miss;
        double hr = total ? (double)hit / total * 100.0 : 0.0;
        LOG_RES("  %-13s : %llu / %llu  (%.2f%%)", name,
                (unsigned long long)hit, (unsigned long long)total, hr);
    };
    log_cache_hit("L1I$ Hit", ri.l1i_hit, ri.l1i_miss);
    log_cache_hit("L1D$ Hit", ri.l1d_hit, ri.l1d_miss);
    /* L1D miss penalty: average stall cycles per L1D miss.
     * Measured by tracking in-flight misses each cycle in tick_posedge(). */
    if (ri.l1d_miss > 0 && ri.l1d_miss_penalty_cycles > 0)
        LOG_RES("  L1D$ Miss/cycle: %.1f stall cycles/miss",
                (double)ri.l1d_miss_penalty_cycles / ri.l1d_miss);
    log_cache_hit("L2$  Hit", ri.l2_hit, ri.l2_miss);
    /* L2 miss penalty: each L2 miss triggers an AXI DRAM fetch. */
    if (ri.l2_miss > 0 && ri.l2_miss_penalty_cycles > 0)
        LOG_RES("  L2$  Miss/cycle: %.1f stall cycles/miss",
                (double)ri.l2_miss_penalty_cycles / ri.l2_miss);

    /* ---- DRAM traffic (L2 -> DRAM) ----
     * Read  = L2 miss refills (cache-line fills from DRAM).
     * Write = L2 dirty evictions (write-back of displaced dirty lines). */
    LOG_RES("  DRAM Read (B) : %u  (L2 miss refills)", ri.memory_read);
    LOG_RES("  DRAM Write(B) : %u  (L2 dirty evictions)", ri.memory_write);
    if (ri.elapsed_time > 0) {
        double bw = (double)(ri.memory_read + ri.memory_write) /
                    ri.elapsed_time * 1000.0; /* B/ns → MB/s */
        LOG_RES("  DRAM BW(MB/s) : %.2f", bw);
    }

    const bool has_case = (case_num >= 0 && case_num <= 5);
    if (has_case)
        LOG_RES("  Errors        : %d  %s", err,
                err == 0 ? "[PASS]" : "[FAIL]");
    LOG_RES("=================================");

    if (has_case) {
        if (err == 0)
            LOG_OK("[TB/CPU] *** TEST PASSED  (cycles=%llu) ***",
                   (unsigned long long)ri.elapsed_cycle);
        else
            LOG_ERR("[TB/CPU] *** FAILED  err=%d ***", err);
    }

    /* ---- Scoring (improve mode only) ----
     * Score = clamp(8 * (orig - improve) / (orig - target), 0, 8)
     */
    if (has_case && use_improve && err == 0) {
        struct {
            uint64_t orig;   /* original cycles */
            uint64_t target; /* target cycles for 100% */
        } scoring_table[5] = {
            /* case0  */ {957120ULL, 580000ULL},
            /* case1  */ {72265534ULL, 33900000ULL},
            /* case2  */ {1004896044ULL, 509500000ULL},
            /* cpu_fallback_linear      */ {23503198ULL, 17300000ULL},
            /* cpu_fallback_linear_relu */ {23505126ULL, 17300000ULL},
        };

        uint64_t orig = scoring_table[case_num].orig;
        uint64_t target = scoring_table[case_num].target;
        uint64_t improve = ri.elapsed_cycle;

        /* Linear interpolation: 0 pts at orig, 8 pts at target */
        double ratio = (orig > target)
                           ? (double)(orig - improve) / (double)(orig - target)
                           : 0.0;
        if (ratio < 0.0) ratio = 0.0;
        if (ratio > 1.0) ratio = 1.0;
        double score = ratio * 8.0;

        printf("\n");
        LOG_RES("===== Scoring (case %d improve) =====", case_num);
        LOG_RES("  Original  cycles : %llu", (unsigned long long)orig);
        LOG_RES("  Target    cycles : %llu  (100%% score)",
                (unsigned long long)target);
        LOG_RES("  Your      cycles : %llu", (unsigned long long)improve);
        LOG_RES("  Reduction ratio  : %.2f%%",
                orig > 0 ? (double)(orig - improve) / orig * 100.0 : 0.0);
        LOG_RES("  Score            : %.2f / 8.0", score);
        LOG_RES("=====================================");
        printf("\n");
    }
    printf("\n");

    return (done && err == 0) ? 0 : 1;
}

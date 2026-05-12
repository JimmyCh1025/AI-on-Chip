// cpu_hal.cpp — CpuHAL implementation

#include "cpu_hal.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>

CpuHAL::CpuHAL(SRAMModel& sram) : sram_(sram) {}

CpuHAL::~CpuHAL() { final(); }

void CpuHAL::init(const char* elf_path, bool enable_trace,
                  const char* trace_path) {
    assert(dut_ == nullptr && "CpuHAL::init called twice");
    dut_ = new VNutShell();

    if (enable_trace) {
        Verilated::traceEverOn(true);
        tfp_ = new VerilatedFstC();
        dut_->trace(tfp_, 99);
        tfp_->open(trace_path);
        fprintf(stdout, "[CPU-HAL] FST trace: %s\n", trace_path);
    }

    sram_.loadELF(elf_path);
    reset_runtime_info();
    fprintf(stdout, "[CPU-HAL] Loaded ELF: %s\n", elf_path);
}

void CpuHAL::reset(int cycles) {
    fprintf(stdout, "[CPU-HAL] Reset %d cycles...\n", cycles);
    dut_->reset = 1;
    dut_->io_mem_aw_ready = 0;
    dut_->io_mem_w_ready = 0;
    dut_->io_mem_b_valid = 0;
    dut_->io_mem_ar_ready = 0;
    dut_->io_mem_r_valid = 0;
    dut_->io_mem_r_bits_data = 0;
    dut_->io_mem_r_bits_last = 0;
    dut_->io_mmio_req_ready = 1;
    dut_->io_mmio_resp_valid = 0;
    dut_->io_mmio_resp_bits_cmd = 0;
    dut_->io_mmio_resp_bits_rdata = 0;
    dut_->io_frontend_aw_ready = 0;
    dut_->io_frontend_w_ready = 0;
    dut_->io_frontend_b_valid = 0;
    dut_->io_frontend_ar_ready = 0;
    dut_->io_frontend_r_valid = 0;
    dut_->io_frontend_r_bits_data = 0;
    dut_->io_meip = 0;

    for (int i = 0; i < cycles; i++) {
        dut_->clock = 0;
        dut_->eval();
        dut_->clock = 1;
        dut_->eval();
        if (tfp_) tfp_->dump((vluint64_t)(info_.elapsed_cycle * 10));
        info_.elapsed_cycle++;
    }
    dut_->reset = 0;
    dut_->eval();
    fprintf(stdout, "[CPU-HAL] Reset done\n\n");
}

void CpuHAL::final() {
    if (tfp_) {
        tfp_->close();
        delete tfp_;
        tfp_ = nullptr;
    }
    if (dut_) {
        delete dut_;
        dut_ = nullptr;
    }
}

void CpuHAL::init() { init(""); }
void CpuHAL::reset() { reset(20); }

/* runtime_info */

struct runtime_info CpuHAL::get_runtime_info() const { return info_; }
struct cpu_runtime_info CpuHAL::get_cpu_runtime_info() const { return info_; }

void CpuHAL::reset_runtime_info() {
    info_ = cpu_runtime_info{};
    l1d_miss_start_cycle_ = -1;
    l2_miss_start_cycle_ = -1;
}

/* Clock primitives — called by tb.cpp run loop each cycle:
 *   1. tick_negedge()       — clock=0, eval, FST
 *   2. [tb.cpp] axi4_tick() — drive io_mem_* pins
 *   3. tick_posedge()       — clock=1, eval, FST, advance counter */

void CpuHAL::tick_negedge() {
    dut_->clock = 0;
    dut_->eval();
    if (tfp_) tfp_->dump((vluint64_t)(info_.elapsed_cycle * 10 + 5));
}

void CpuHAL::tick_posedge() {
    dut_->clock = 1;
    dut_->eval();
    if (tfp_) tfp_->dump((vluint64_t)((info_.elapsed_cycle + 1) * 10));

    /* --- Cache hit / miss counters ---
     * Raw RTL hit/miss signals are level-based and cause overcounting.
     * To count exactly once, we trigger on the 1-cycle `isFinish` pulse
     * to check `s3_io_in_bits_r_hit`. L1D/L2 coherence probes are skipped. */

    /* L1 I-Cache */
    {
        bool l1i_finish =
            dut_->rootp
                ->NutShell__DOT__nutcore__DOT__io_imem_cache__DOT___s3_io_isFinish;
        bool l1i_was_hit =
            dut_->rootp
                ->NutShell__DOT__nutcore__DOT__io_imem_cache__DOT__s3_io_in_bits_r_hit;
        if (l1i_finish) {
            if (l1i_was_hit)
                info_.l1i_hit++;
            else
                info_.l1i_miss++;
        }
    }

    /* L1 D-Cache (skip probe) */
    {
        bool l1d_finish =
            dut_->rootp
                ->NutShell__DOT__nutcore__DOT__io_dmem_cache__DOT___s3_io_isFinish;
        bool l1d_was_hit =
            dut_->rootp
                ->NutShell__DOT__nutcore__DOT__io_dmem_cache__DOT__s3_io_in_bits_r_hit;
        bool l1d_probe =
            dut_->rootp
                ->NutShell__DOT__nutcore__DOT__io_dmem_cache__DOT__s3__DOT__probe;
        if (l1d_finish && !l1d_probe) {
            if (l1d_was_hit)
                info_.l1d_hit++;
            else
                info_.l1d_miss++;
        }
    }

    /* L2 Cache (skip probe) */
    {
        bool l2_finish =
            dut_->rootp
                ->NutShell__DOT__mem_l2cacheOut_cache__DOT___s3_io_isFinish;
        bool l2_was_hit =
            dut_->rootp
                ->NutShell__DOT__mem_l2cacheOut_cache__DOT__s3_io_in_bits_r_hit;
        bool l2_probe =
            dut_->rootp
                ->NutShell__DOT__mem_l2cacheOut_cache__DOT__s3__DOT__probe;
        if (l2_finish && !l2_probe) {
            if (l2_was_hit)
                info_.l2_hit++;
            else
                info_.l2_miss++;
        }
    }

    info_.elapsed_cycle++;
    info_.elapsed_time += CYCLE_TIME;

    /* --- Hardware Signals for Miss Penalty Estimation --- */
    bool l1d_miss_issued =
        dut_->rootp
            ->NutShell__DOT__nutcore__DOT__io_dmem_cache__DOT__s3__DOT__miss;
    bool l1d_data_return =
        dut_->rootp
            ->NutShell__DOT__nutcore__DOT__io_dmem_cache__DOT__s3__DOT__hit;

    bool l2_miss_issued =
        dut_->rootp->NutShell__DOT__mem_l2cacheOut_cache__DOT__s3__DOT__miss;
    bool l2_data_return =
        dut_->rootp->NutShell__DOT__mem_l2cacheOut_cache__DOT__s3__DOT__hit;


    // L1D miss penalty
    if (l1d_miss_issued && (l1d_miss_start_cycle_ == -1)) 
    {
        l1d_miss_start_cycle_ = info_.elapsed_cycle;
    }

    if (l1d_data_return && (l1d_miss_start_cycle_ != -1)) 
    {
        info_.l1d_miss_penalty_cycles += (info_.elapsed_cycle - l1d_miss_start_cycle_);
        l1d_miss_start_cycle_ = -1;
    }

    // L2 miss penalty
    if (l2_miss_issued && (l2_miss_start_cycle_ == -1)) 
    {
        l2_miss_start_cycle_ = info_.elapsed_cycle;
    }

    if (l2_data_return && (l2_miss_start_cycle_ != -1)) 
    {
        info_.l2_miss_penalty_cycles += (info_.elapsed_cycle - l2_miss_start_cycle_);
        l2_miss_start_cycle_ = -1;
    }

}

void CpuHAL::stop_trace() {
    if (tfp_) {
        tfp_->close();
        delete tfp_;
        tfp_ = nullptr;
        fprintf(stdout, "[CPU-HAL] Waveform stopped at cycle %llu\n",
                (unsigned long long)info_.elapsed_cycle);
    }
}

void CpuHAL::update_memory_stats(uint32_t rd, uint32_t wr) {
    info_.memory_read += rd;
    info_.memory_write += wr;
}

/* Done / halt detection */

bool CpuHAL::check_done() {
    static const uint32_t DONE_FLAG_ADDR = 0x80000004u;
    static const uint32_t DONE_MAGIC = 0xDEADu;
    uint64_t flag64 = peek_dcache(DONE_FLAG_ADDR & ~7u);
    uint32_t flag = (uint32_t)(flag64 >> ((DONE_FLAG_ADDR & 4u) * 8u));
    return flag == DONE_MAGIC;
}

bool CpuHAL::check_halted() {
    uint64_t cur_mepc =
        dut_->rootp
            ->NutShell__DOT__nutcore__DOT__backend__DOT__exu__DOT__csr__DOT__mepc;
    if (cur_mepc == 0) return false;
    if (cur_mepc == halt_prev_mepc_) {
        if (++halt_count_ >= 5) return true;
    } else {
        halt_count_ = 0;
        halt_prev_mepc_ = cur_mepc;
    }
    return false;
}

uint64_t CpuHAL::peek_dcache(uint32_t addr) {
    uint64_t sram_val = sram_.peek(addr);
    uint32_t set_idx = (addr >> 6) & 0x7Fu;
    uint32_t word_off = (addr >> 3) & 0x7u;
    uint32_t addr_tag = (addr >> 13) & 0x7FFFFu;

    auto& meta_entry =
        dut_->rootp
            ->NutShell__DOT__nutcore__DOT__io_dmem_cache__DOT__metaArray__DOT__ram__DOT__array_ext__DOT__Memory
                [set_idx];
    uint64_t meta_lo =
        (uint64_t)meta_entry.at(0) | ((uint64_t)meta_entry.at(1) << 32);
    uint32_t meta_hi = meta_entry.at(2);

    uint32_t data_addr = set_idx * 8u + word_off;
    auto& data_entry =
        dut_->rootp
            ->NutShell__DOT__nutcore__DOT__io_dmem_cache__DOT__dataArray__DOT__ram__DOT__array_ext__DOT__Memory
                [data_addr];

    for (int way = 0; way < 4; way++) {
        uint32_t meta21;
        switch (way) {
            case 0:
                meta21 = (uint32_t)(meta_lo & 0x1FFFFFu);
                break;
            case 1:
                meta21 = (uint32_t)((meta_lo >> 21) & 0x1FFFFFu);
                break;
            case 2:
                meta21 = (uint32_t)((meta_lo >> 42) & 0x1FFFFFu);
                break;
            default:
                meta21 =
                    (uint32_t)(((meta_lo >> 63) | ((uint64_t)meta_hi << 1)) &
                               0x1FFFFFu);
                break;
        }
        uint32_t way_valid = (meta21 >> 1) & 1u;
        uint32_t way_tag = (meta21 >> 2) & 0x7FFFFu;
        if (!way_valid || way_tag != addr_tag) continue;

        return (uint64_t)data_entry.at(way * 2) |
               ((uint64_t)data_entry.at(way * 2 + 1) << 32);
    }
    return sram_val;
}

/* Flush dirty lines to SRAM. L2 first, then L1 (L1 wins on conflict). */
void CpuHAL::sync_dcache_to_sram() {
    /* L2: 512 sets × 4 ways, 19 bits/way
     * Meta: {tag[16:0], valid, dirty}  bit0=dirty, bit1=valid, bits[18:2]=tag
     * Data: array_4096x256, idx=set*8+word, 256 bits = 4 ways × 64 bits
     * Addr: tag=addr[31:15], set=addr[14:6] */
    for (uint32_t set = 0; set < 512; set++) {
        auto& meta_entry =
            dut_->rootp
                ->NutShell__DOT__mem_l2cacheOut_cache__DOT__metaArray__DOT__ram__DOT__array_ext__DOT__Memory
                    [set];
        uint64_t meta_lo =
            (uint64_t)meta_entry.at(0) | ((uint64_t)meta_entry.at(1) << 32);
        uint32_t meta_hi = meta_entry.at(2);

        for (int way = 0; way < 4; way++) {
            uint32_t meta19;
            switch (way) {
                case 0:
                    meta19 = (uint32_t)(meta_lo & 0x7FFFFu);
                    break;
                case 1:
                    meta19 = (uint32_t)((meta_lo >> 19) & 0x7FFFFu);
                    break;
                case 2:
                    meta19 = (uint32_t)((meta_lo >> 38) & 0x7FFFFu);
                    break;
                default:
                    meta19 = (uint32_t)(((meta_lo >> 57) |
                                         ((uint64_t)meta_hi << 7)) &
                                        0x7FFFFu);
                    break;
            }
            uint32_t way_dirty = meta19 & 1u;
            uint32_t way_valid = (meta19 >> 1) & 1u;
            uint32_t way_tag = (meta19 >> 2) & 0x1FFFFu;
            if (!way_valid || !way_dirty) continue;

            uint32_t cache_addr = ((uint32_t)way_tag << 15) | (set << 6);
            for (int word_idx = 0; word_idx < 8; word_idx++) {
                auto& data_entry =
                    dut_->rootp
                        ->NutShell__DOT__mem_l2cacheOut_cache__DOT__dataArray__DOT__ram__DOT__array_ext__DOT__Memory
                            [set * 8 + word_idx];
                uint64_t data = (uint64_t)data_entry.at(way * 2) |
                                ((uint64_t)data_entry.at(way * 2 + 1) << 32);
                sram_.write(cache_addr + word_idx * 8, data, 0xFF);
            }
        }
    }

    /* L1 D-Cache: 128 sets × 4 ways, 21 bits/way
     * Meta: {tag[18:0], valid, dirty}  Addr: tag=addr[31:13], set=addr[12:6] */
    for (uint32_t set = 0; set < 128; set++) {
        auto& meta_entry =
            dut_->rootp
                ->NutShell__DOT__nutcore__DOT__io_dmem_cache__DOT__metaArray__DOT__ram__DOT__array_ext__DOT__Memory
                    [set];
        uint64_t meta_lo =
            (uint64_t)meta_entry.at(0) | ((uint64_t)meta_entry.at(1) << 32);
        uint32_t meta_hi = meta_entry.at(2);

        for (int way = 0; way < 4; way++) {
            uint32_t meta21;
            switch (way) {
                case 0:
                    meta21 = (uint32_t)(meta_lo & 0x1FFFFFu);
                    break;
                case 1:
                    meta21 = (uint32_t)((meta_lo >> 21) & 0x1FFFFFu);
                    break;
                case 2:
                    meta21 = (uint32_t)((meta_lo >> 42) & 0x1FFFFFu);
                    break;
                default:
                    meta21 = (uint32_t)(((meta_lo >> 63) |
                                         ((uint64_t)meta_hi << 1)) &
                                        0x1FFFFFu);
                    break;
            }
            uint32_t way_dirty = meta21 & 1u;
            uint32_t way_valid = (meta21 >> 1) & 1u;
            uint32_t way_tag = (meta21 >> 2) & 0x7FFFFu;
            if (!way_valid || !way_dirty) continue;

            uint32_t cache_addr = ((uint32_t)way_tag << 13) | (set << 6);
            for (int word_idx = 0; word_idx < 8; word_idx++) {
                auto& data_entry =
                    dut_->rootp
                        ->NutShell__DOT__nutcore__DOT__io_dmem_cache__DOT__dataArray__DOT__ram__DOT__array_ext__DOT__Memory
                            [set * 8 + word_idx];
                uint64_t data = (uint64_t)data_entry.at(way * 2) |
                                ((uint64_t)data_entry.at(way * 2 + 1) << 32);
                sram_.write(cache_addr + word_idx * 8, data, 0xFF);
            }
        }
    }
}

uint32_t CpuHAL::get_symbol_addr(const char* name) {
    auto it = sram_.symbols.find(std::string(name));
    if (it == sram_.symbols.end()) {
        fprintf(stderr, "[CpuHAL] WARNING: symbol \"%s\" not found in ELF\n",
                name);
        return 0u;
    }
    return (uint32_t)it->second;
}

void CpuHAL::read_output(uint32_t addr, uint8_t* buf, uint32_t len) {
    sync_dcache_to_sram();
    for (uint32_t i = 0; i < len; i++) {
        uint32_t byte_addr = addr + i;
        uint64_t word = sram_.peek(byte_addr & ~7u);
        buf[i] = (uint8_t)(word >> ((byte_addr & 7u) * 8u));
    }
}

/* External symbol stubs required by NutShell Verilator model */
extern "C" {
void xs_assert(long long) {}
void xs_assert_v2(const char*, long long) {}
void sd_setaddr(int) {}
void sd_read(int*) {}
void put_pixel(int) {}
void vmem_sync() {}
long long difftest_ram_read(long long) { return 0LL; }
void difftest_ram_write(long long, long long, long long) {}
}

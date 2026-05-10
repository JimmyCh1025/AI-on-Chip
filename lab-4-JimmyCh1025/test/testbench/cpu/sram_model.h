// sram_model.h — Sparse AXI4 SRAM model (64-bit aligned).

#ifndef SRAM_MODEL_H
#define SRAM_MODEL_H

#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>

#include "elf_loader.h"

class SRAMModel {
   public:
    /* Sparse memory keyed by 8-byte aligned address */
    std::map<uint32_t, uint64_t> mem;
    /* ELF symbol name -> virtual address */
    std::map<std::string, uint64_t> symbols;

    uint64_t read(uint32_t addr) {
        addr &= ~7u; /* align to 8 bytes */
        auto it = mem.find(addr);
        return (it != mem.end()) ? it->second : 0ULL;
    }

    void write(uint32_t addr, uint64_t data, uint8_t strb) {
        addr &= ~7u;
        uint64_t old = read(addr);
        uint64_t mask = 0;
        for (int i = 0; i < 8; i++) {
            if (strb & (1u << i)) mask |= (0xFFULL << (i * 8));
        }
        mem[addr] = (old & ~mask) | (data & mask);
    }

    void loadELF(const char* filename) {
        ELFLoader::LoadedELF elf = ELFLoader::loadFromFile(filename);
        for (auto& seg : elf.segments) {
            printf("[SRAM] segment: paddr=0x%08lx size=%zu\n",
                   (unsigned long)seg.paddr, seg.file_size);
            for (size_t off = 0; off < seg.mem_size; off++) {
                uint32_t byte_addr = (uint32_t)(seg.paddr + off);
                uint32_t word_addr = byte_addr & ~7u;
                int byte_off = byte_addr & 7;
                uint8_t byte_val = (off < seg.file_size) ? seg.data[off] : 0;
                uint64_t cur = read(word_addr);
                uint64_t mask = 0xFFULL << (byte_off * 8);
                mem[word_addr] =
                    (cur & ~mask) | ((uint64_t)byte_val << (byte_off * 8));
            }
        }
        printf("[SRAM] ELF loaded, entry=0x%08lx, %zu segments\n",
               (unsigned long)elf.entry_point, elf.segments.size());
        symbols = elf.symbols;
    }

    /* Direct byte write for testbench pre-injection. */
    void write_bytes(uint32_t addr, const uint8_t* buf, uint32_t len) {
        for (uint32_t i = 0; i < len; i++) {
            uint32_t byte_addr = addr + i;
            uint32_t word_addr = byte_addr & ~7u;
            int byte_off = (int)(byte_addr & 7u);
            uint64_t cur = read(word_addr);
            uint64_t mask = 0xFFULL << (byte_off * 8);
            mem[word_addr] =
                (cur & ~mask) | ((uint64_t)buf[i] << (byte_off * 8));
        }
    }

    uint64_t peek(uint32_t addr) { return read(addr); }
    uint32_t peek32(uint32_t addr) {
        uint64_t v = read(addr & ~7u);
        return (uint32_t)(v >> ((addr & 4) * 8));
    }

    void dump(uint32_t start, uint32_t end) {
        printf("[SRAM] dump [0x%08x .. 0x%08x]\n", start, end);
        for (uint32_t addr = start & ~7u; addr <= end; addr += 8) {
            auto it = mem.find(addr);
            if (it != mem.end()) {
                printf("  [0x%08x] = 0x%016llx\n", addr,
                       (unsigned long long)it->second);
            }
        }
    }
};

/* AXI4 Read Slave SM */
struct AXI4ReadSM {
    enum State { IDLE, DELAY, ACTIVE } state = IDLE;
    uint32_t base_addr = 0;
    uint8_t len = 0;
    uint8_t beat = 0;

    bool arready() const { return state == IDLE; }

    void accept(uint32_t a, uint8_t l) {
        base_addr = a & ~7u;
        len = l;
        beat = 0;
        state = DELAY;
    }

    void tick_delay() {
        if (state == DELAY) state = ACTIVE;
    }

    void advance() {
        beat++;
        if (beat > len) state = IDLE;
    }

    bool valid() const { return state == ACTIVE; }
    bool is_last() const { return beat == len; }
    uint32_t data_addr() const {
        /* WRAP burst: wrap within (len+1)*8 bytes from aligned boundary */
        uint32_t wrap_size = (uint32_t)(len + 1) * 8;
        uint32_t wrap_mask = wrap_size - 1;
        uint32_t aligned = base_addr & ~wrap_mask;
        uint32_t offset = ((base_addr & wrap_mask) + beat * 8) & wrap_mask;
        return aligned | offset;
    }
};

/* AXI4 Write Slave SM */
struct AXI4WriteSM {
    enum State { IDLE, AWAIT_DATA, DELAY_RESP, AWAIT_RESP } state = IDLE;
    uint32_t addr = 0;
    uint8_t len = 0;
    uint8_t beat = 0;

    bool awready() const { return state == IDLE; }
    bool wready() const { return state == AWAIT_DATA; }
    bool bvalid() const { return state == AWAIT_RESP; }

    void tick_delay() {
        if (state == DELAY_RESP) state = AWAIT_RESP;
    }

    void accept_addr(uint32_t a, uint8_t l) {
        addr = a & ~7u;
        len = l;
        beat = 0;
        state = AWAIT_DATA;
    }

    uint32_t data_addr() const {
        /* WRAP burst: wrap within (len+1)*8 bytes from aligned boundary */
        uint32_t wrap_size = (uint32_t)(len + 1) * 8;
        uint32_t wrap_mask = wrap_size - 1;
        uint32_t aligned = addr & ~wrap_mask;
        uint32_t offset = ((addr & wrap_mask) + beat * 8) & wrap_mask;
        return aligned | offset;
    }

    void accept_data(uint64_t data, uint8_t strb, bool wlast, SRAMModel& sram) {
        sram.write(data_addr(), data, strb);
        beat++;
        if (wlast || beat > len) state = DELAY_RESP;
    }

    void accept_resp() { state = IDLE; }
};

#endif  // SRAM_MODEL_H

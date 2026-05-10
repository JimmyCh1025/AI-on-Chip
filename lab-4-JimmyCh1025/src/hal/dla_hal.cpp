// dla_hal.cpp — DlaHAL implementation

#include "dla_hal.hpp"

#include <cstdio>
#include <cstring>

#ifdef USE_FST
void DlaHAL::fst_init() {
    Verilated::traceEverOn(true);
    FST_FP = new VerilatedFstC();
    device_->trace(FST_FP, DLA_TRACE_DEPTH);
    fprintf(stdout, "[DLA-HAL] FST trace enabled\n");
}

void DlaHAL::fst_final() {
    if (FST_FP) {
        delete FST_FP;
        FST_FP = nullptr;
    }
}
#endif

DlaHAL::DlaHAL(uint32_t baseaddr, uint32_t mmio_size)
    : info_{},
      baseaddr_(baseaddr),
      mmio_size_(mmio_size),
      device_(nullptr),
      vm_addr_h_(0) {
    vm_addr_h_ = (reinterpret_cast<uint64_t>(this) & 0xffffffff00000000ULL);
#ifdef DEBUG
    fprintf(stderr, "[DLA-HAL] vm_addr_h = 0x%lx\n", (unsigned long)vm_addr_h_);
#endif
#ifdef USE_FST
    fst_task_id_ = 0;
#endif
    device_ = new Vasic_wrapper("TOP");
}

DlaHAL::~DlaHAL() {
    if (device_) {
        delete device_;
        device_ = nullptr;
    }
#ifdef DEBUG
    fprintf(stderr, "[DLA-HAL] destroyed\n");
#endif
}

void DlaHAL::init() {
#ifdef USE_FST
    fst_init();
#endif
#ifdef DEBUG
    fprintf(stderr, "[DLA-HAL] init\n");
#endif
    reset_runtime_info();
    reset();
}

void DlaHAL::reset() {
    device_->ARESETn = 0;
    for (uint32_t i = 0; i < DLA_RESET_CYCLE; i++) {
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    }
    device_->ARESETn = 1;
    device_->eval();
}

void DlaHAL::final() {
#ifdef USE_FST
    fst_final();
#endif
}

struct runtime_info DlaHAL::get_runtime_info() const { return info_; }

void DlaHAL::reset_runtime_info() {
    info_.elapsed_cycle = 0;
    info_.elapsed_time = 0;
    info_.memory_read = 0;
    info_.memory_write = 0;
}

/* MMIO write (AXI Slave Write) */
bool DlaHAL::memory_set(uint32_t addr, uint32_t data) {
    if (!device_) {
        fprintf(stderr, "[DLA-HAL] device not initialised\n");
        return false;
    }

#ifdef DEBUG
    fprintf(stderr, "[DLA-HAL] memory_set(0x%08x) = 0x%08x\n", addr, data);
#endif
    if (addr < baseaddr_ || addr >= baseaddr_ + mmio_size_) {
#ifdef DEBUG
        fprintf(stderr, "[DLA-HAL] address 0x%08x out of MMIO range\n", addr);
#endif
        return false;
    }

    /* AW channel */
    // send write address
    device_->AWID_S = 0;
    device_->AWADDR_S = addr;
    device_->AWLEN_S = 0;    // unused
    device_->AWSIZE_S = 0;   // unused
    device_->AWBURST_S = 0;  // unused
    device_->AWVALID_S = 1;  // valid
    device_->eval();

    // wait for ready (address)
    while (!device_->AWREADY_S) {
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    }
    clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    device_->AWVALID_S = 0;


    /* W channel */
    // send write data
    device_->WDATA_S = data;
    device_->WSTRB_S = 0;   // unused
    device_->WLAST_S = 1;   // single shot, always the last one
    device_->WVALID_S = 1;  // valid
    device_->eval();

    // wait for ready (data)
    while (!device_->WREADY_S) {
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    }
    clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    device_->WVALID_S = 0;


    // wait for write response
    device_->BREADY_S = 1;
    device_->eval();
    while (!device_->BVALID_S) {
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    }
    clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    device_->BREADY_S = 0;


    int resp = device_->BRESP_S;
    clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    return resp == AXI_RESP_OKAY;
}

/* MMIO read (AXI Slave Read) */
bool DlaHAL::memory_get(uint32_t addr, uint32_t& data) {
    if (!device_) {
        fprintf(stderr, "[DLA-HAL] device not initialised\n");
        return false;
    }

#ifdef DEBUG
    fprintf(stderr, "[DLA-HAL] memory_get(0x%08x)\n", addr);
#endif
    if (addr < baseaddr_ || addr >= baseaddr_ + mmio_size_) {
#ifdef DEBUG
        fprintf(stderr, "[DLA-HAL] address 0x%08x out of MMIO range\n", addr);
#endif
        return false;
    }

    /* AR channel */
    // send read address
    device_->ARID_S = 0;
    device_->ARADDR_S = addr;
    device_->ARLEN_S = 0;    // unused
    device_->ARSIZE_S = 0;   // unused
    device_->ARBURST_S = 0;  // unused
    device_->ARVALID_S = 1;  // valid

    // wait for ready (address)
    do {
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    } while (!device_->ARREADY_S);
    device_->ARVALID_S = 0;


    /* R channel */
    // wait for valid (data)
    device_->RREADY_S = 1;
    do {
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    } while (!device_->RVALID_S);
    device_->RREADY_S = 0;

    // get read data
    data = device_->RDATA_S;
    int resp = device_->RRESP_S;
    clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    return resp == AXI_RESP_OKAY;
}

/* Block until DLA interrupt */
void DlaHAL::wait_for_irq() {
    if (!device_) {
        fprintf(stderr, "[DLA-HAL] device not initialised\n");
        return;
    }

#ifdef DEBUG
    fprintf(stderr, "[DLA-HAL] wait_for_irq\n");
#endif

#ifdef USE_FST
#ifndef DLA_FST_DIR
#define DLA_FST_DIR ""
#endif
    char filename[256];
    snprintf(filename, sizeof(filename), "%sasic_%d.fst", DLA_FST_DIR,
             fst_task_id_);
    FST_FP->open(filename);
#endif

    while (!device_->ASIC_interrupt) {
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
        if (device_->ARVALID_M) handle_dma_read();
        if (device_->AWVALID_M) handle_dma_write();
    }

#ifdef USE_FST
    FST_FP->close();
    fst_task_id_++;
#endif
}

/* DMA read — DLA requests data from host memory */
void DlaHAL::handle_dma_read() {
    uint32_t* addr =
        reinterpret_cast<uint32_t*>(vm_addr_h_ | device_->ARADDR_M);
    uint32_t len = device_->ARLEN_M;

    device_->ARREADY_M = 1;
    clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    device_->ARREADY_M = 0;
    clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);

#ifdef DEBUG
    fprintf(stderr, "[DLA-HAL] DMA read addr=%p len=%u\n", addr, len + 1);
#endif

    device_->RID_M = 0;
    device_->RRESP_M = AXI_RESP_OKAY;

    // send read data (increase mode, burst_size 32bits)
    device_->RID_M = 0;  // default
    device_->RRESP_M = AXI_RESP_OKAY;

    for (int i = 0; i <= len; i++) {
        device_->RDATA_M = *(addr + i);            // send read data
        info_.elapsed_cycle += MEM_ACCESS_CYCLE;   // simulate memory access delay
        info_.elapsed_time  += MEM_ACCESS_CYCLE * CYCLE_TIME;

#ifdef DEBUG
        fprintf(stdout, "[DLA-HAL] DMA R addr=%p data=0x%08x\n", addr + i,
                *(addr + i));
#endif
        device_->RLAST_M = i == len;  // the last one
        device_->RVALID_M = 1;
        device_->eval();

        // wait DMA ready for next data
        while (!device_->RREADY_M) {
            clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
        }
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
        device_->RVALID_M = 0;
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    }
    device_->eval();


    info_.memory_read += sizeof(uint32_t) * (len + 1);
}

/* DMA write — DLA writes data to host memory */
void DlaHAL::handle_dma_write() {
    uint32_t* addr =
        reinterpret_cast<uint32_t*>(vm_addr_h_ | device_->AWADDR_M);
    uint32_t len = device_->AWLEN_M;

    device_->AWREADY_M = 1;
    clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    device_->AWREADY_M = 0;
    clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);

#ifdef DEBUG
    fprintf(stderr, "[DLA-HAL] DMA write addr=%p len=%u\n", addr, len + 1);
#endif

    /* W channel */
    // recv write data (increase mode, burst_size 32bits)
    for (uint32_t i = 0; i <= len; i++) {
        *(addr + i) = static_cast<uint32_t>(device_->WDATA_M);  // recv write data
        info_.elapsed_cycle += MEM_ACCESS_CYCLE;   // simulate memory access delay
        info_.elapsed_time  += MEM_ACCESS_CYCLE * CYCLE_TIME;

#ifdef DEBUG
        fprintf(stdout, "[DLA-HAL] DMA W addr=%p data=0x%08x\n", addr + i,
                *(addr + i));
#endif
        device_->WREADY_M = 1;
        device_->eval();

        // wait DMA valid for next data
        while (!device_->WVALID_M) {
            clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
        }
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
        device_->WREADY_M = 0;
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    }
    device_->eval();


    /* B channel */
    // recv write response
    device_->BID_M = 0;
    device_->BRESP_M = AXI_RESP_OKAY;
    device_->BVALID_M = 1;
    device_->eval();
    while (!device_->BREADY_M) {
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    };
    clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    device_->BVALID_M = 0;
    device_->eval();


    info_.memory_write += sizeof(uint32_t) * (len + 1);
}

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/WP9Xe0FP)
# Lab 4: Runtime and Performance Profiling

This document explains how to set up the environment, build the hardware simulation libraries, run workload benchmarks, and submit your assignment for **Lab 4**.

---

## 1. Project Overview

Lab 4 compares the execution cycle counts of the same quantized convolution workloads on two accelerator backends:

| Backend | Description |
|---------|-------------|
| **DLA** | Custom Eyeriss-like ASIC, simulated via Verilator |
| **CPU** | NutShell RV64IM soft-core, simulated via Verilator. Split into `original` and `improve` versions |

---

## 2. Quick Start

### View Available Targets

```bash
make help
```

This prints all available `make` targets and options, including:

- **Options**: `IMPROVE=1`, `DIAG=1` for CPU; `INFO=1`, `DEBUG=1` for DLA
- **Run targets**: individual (`run_cpu_caseN`, `run_dla_caseN`) and batch (`run_cpu`, `run_dla`, `run_all`)
- **Clean targets**: granular clean per component (`clean_hw_cpu`, `clean_runtime_dla`, etc.)

### Build Everything at Once

```bash
make all
```
This builds both hardware simulation libraries and all runtime ELFs for CPU and DLA backends. The first build may take several minutes due to the large NutShell RTL.

---

## 3. Running Tests

### 3.1 DLA Backend (DLA RTL Simulation)

```bash
# Run all DLA cases
make run_dla

# Run a specific case
make run_dla_case0
make run_dla_case1
make run_dla_case2
```

**Available Options:**

| Option | Description |
|--------|-------------|
| `INFO=1` | Dump DLA statistics to `build/dla_info_caseN.csv` |
| `DEBUG=1` | Print DLA HAL verbose log (MMIO R/W, DMA R/W addr+data, IRQ) |

### 3.2 CPU Backend (NutShell RTL Simulation)

```bash
# Run all CPU cases — both original and improve
make run_cpu

# Run all CPU cases — original only
make run_cpu_original

# Run all CPU cases — improve only
make run_cpu_improve

# Run a specific case (original by default)
make run_cpu_case0
make run_cpu_case1 IMPROVE=1

# Run CPU fallback cases (linear / linear_relu)
make run_cpu_fallback_linear
make run_cpu_fallback_linear_relu
```

**Available Options:**

| Option | Description |
|--------|-------------|
| `IMPROVE=1` | Select the `improve` runtime variant (default: `original`) |
| `DIAG=1` | Print the first 16 element-wise mismatches on failure |

**Sample Output:**

```
===== CPU Simulation Result =====
  Case          : case0 original
  Cycles        : 957120
  Time (s)      : 0.004786
  L1I$ Hit      : 636963 / 636987  (100.00%)
  L1D$ Hit      : 71187 / 71225  (99.95%)
  L2$  Hit      : 54 / 131  (41.22%)
  DRAM Read (B) : 4928  (L2 miss refills)
  DRAM Write(B) : 0  (L2 dirty evictions)
  DRAM BW(MB/s) : 1.03
  Errors        : 0  [PASS]
=================================
[TB/CPU] *** TEST PASSED  (cycles=957120) ***
```

### 3.3 Run Everything

```bash
make run_all            # Run all CPU cases (improve) + all DLA cases
```

---

## 4. Clean Build Artifacts

```bash
make clean              # Clean everything
make clean_hw           # Clean both hardware Verilated libs
make clean_hw_cpu       # Clean CPU (NutShell) Verilated lib only
make clean_hw_dla       # Clean DLA Verilated lib only
make clean_runtime      # Clean all runtime ELFs + testbench artifacts
make clean_runtime_cpu  # Clean CPU runtime ELFs + CPU testbench
make clean_runtime_dla  # Clean DLA testbench artifacts
```

---

## 5. Submission Guidelines

### Deadline

- **May 18, 2026 (Monday) at 23:59:59**
- **Late submissions will not be accepted**

---

### Required Submission Structure

```
$PROJECT_ROOT/
├── include/
│   ├── hal/
│   │   ├── hal.hpp                 // Abstract HAL interface
│   │   ├── cpu_hal.hpp             // CPU HAL class definition
│   │   └── dla_hal.hpp             // DLA HAL class definition
│   └── runtime/
│       └── runtime.h               // Public DLA + CPU runtime API
├── src/
│   ├── hal/
│   │   ├── cpu_hal.cpp             // CPU HAL implementation
│   │   └── dla_hal.cpp             // DLA HAL implementation
│   ├── runtime/
│   │   ├── cpu/
│   │   │   ├── original/
│   │   │   │   ├── kernel_cpu.h    // Reference kernel prototypes
│   │   │   │   ├── kernel_cpu.c    // Reference kernel implementations
│   │   │   │   └── runtime_cpu.c   // Glue layer (weight repack + dispatch)
│   │   │   └── improve/
│   │   │       ├── kernel_cpu.h    // Optimized kernel prototypes
│   │   │       ├── kernel_cpu.c    // Cache-optimized kernel implementations
│   │   │       └── runtime_cpu.c   // Glue layer + scratch buffer
│   │   └── dla/
│   │       ├── driver_dla.h        // DLA MMIO register map + register write API
│   │       ├── driver_dla.cpp      // DLA register setting functions
│   │       └── runtime_dla.cpp     // High-level DLA operation API (qconv2d_relu, etc.)
│   └── hardware/
│       ├── cpu/                    // NutShell RTL source files
│       └── dla/                    // DLA RTL source files
├── test/
│   ├── cases/
│   │   ├── case0/                  // Test: Conv + ReLU
│   │   ├── case1/                  // Test: Conv + ReLU + MaxPool
│   │   ├── case2/                  // Test: Conv + ReLU + MaxPool (larger)
│   │   └── case_cpu_fallback/      // Test: Linear / Linear+ReLU
│   └── testbench/
│       ├── cpu/                    // CPU testbench with NutShell Verilator model
│       └── dla/                    // DLA testbench with DLA Verilator model
├── report.md                       // Submission report template
└── Makefile                        // Top-level build targets
```

> **DO NOT modify** `Makefile`, `grade.py` and files under `src/hardware/`, `test/` or `.github/`.

---

### How to Submit via GitHub Classroom

#### Step 1: Check Your Changes

```bash
git status
```

#### Step 2: Add Files to Commit

```bash
git add .
```

#### Step 3: Commit Your Work

```bash
git commit -m "improve: add scratch buffer tiling for case1"
```

You may commit multiple times before the deadline.

#### Step 4: Push to GitHub Classroom

```bash
git push origin main
```

#### Step 5: Verify Submission

1. Go to your **GitHub Classroom repository page**.
2. Confirm the latest commit is pushed.
3. Ensure:
   - Required files exist
   - No changes in `Makefile`, `grade.py` and files under `src/hardware`, `test/` or `.github/`
4. Wait a moment for autograding results.

---

### Notes

- You can push multiple times; **the latest commit before the deadline is graded**.
- If you forget to push, your work is **not submitted**.
- Always verify on GitHub after pushing.

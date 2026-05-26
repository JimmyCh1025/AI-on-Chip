[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/SQ1tkQe3)
# Lab 5 - AI Compiler

[Lab 5 - AI Compiler](https://hackmd.io/@aoc-2026spring/2026-aoc-lab5)

This README is a quick entry point for the 2026 Lab 5 workflow. Read the HackMD document linked above for the full teaching content, and fill [`report.md`](./report.md) for homework submission.

All student-facing commands below should be run from the top-level `lab-5/` directory. The subdirectory Makefiles are implementation details wrapped by the top-level Makefile.

## Getting Started

In this lab, using [AoC Workspace](https://github.com/AI-on-Chip-at-NCKU-EE/aoc-workspace) is required.

1. Accept the assignment from GitHub Classroom.
2. Clone your repository:

    ```bash
    git clone git@github.com:AI-on-Chip-at-NCKU-EE/lab-5-<your-id>.git
    cd lab-5-<your-id>
    ```

3. If the course container does not include `dlpack.h`, patch it inside the container:

    ```bash
    mkdir -p ~/tvm/include/dlpack
    wget https://raw.githubusercontent.com/dmlc/dlpack/v0.8/include/dlpack/dlpack.h -O ~/tvm/include/dlpack/dlpack.h
    ```

4. Valgrind Massif and `ms_print` are available inside the course container. Install `massif-visualizer` only if you want to inspect Massif output with a GUI on a Linux or WSL host.

## Top-Level Workflow

Use `make help` whenever you forget the available targets:

```bash
make help
```

For a quick development sanity check:

```bash
make all
```

`make all` builds the TVM artifacts, runs one CPU-only inference, runs one DLA inference, and generates Relay graph SVGs. It is useful while developing, but it does not generate every artifact required by the report.

For the complete homework artifact flow:

```bash
make homework
```

`make homework` is the canonical report workflow. It builds the model, strictly generates both Relay graph visualizations, runs the CPU-only 100-image accuracy test, profiles one CPU-only inference with Massif, creates an `ms_print` text report, and runs DLA simulation with per-layer statistics enabled.

## Build and Visualize

Generate TVM libraries, Relay dumps, and `input.bin`:

```bash
make build_model
```

Generate the before-fusion Relay graph, then best-effort try the after-fusion graph:

```bash
make visuTVM
```

If the fusion patterns are not finished yet, `make visuTVM` still leaves the before-fusion graph usable and prints a hint. You can also generate each graph explicitly:

```bash
make visuTVM_origin
make visuTVM_pass
```

Expected Relay graph artifacts:

- `output/visu_VGG8_relay_ir.svg` from `make visuTVM` or `make visuTVM_origin`
- `output/visu_VGG8_relay_ir_pass.svg` from `make visuTVM_pass` after fusion is complete

The generated weights are embedded into the TVM-generated C/shared-library artifacts. The testbenches use `output/bin/input.bin` and link against `output/lib_cpu.so` or `output/lib_dla.so`.

## CPU-Only Flow

Run one CPU-only smoke test:

```bash
make test_cpu
```

Run the 100-image CPU-only accuracy test for the report:

```bash
make test_cpu_full
```

The full-test log is written to:

```text
testbench/cpu/log/out.log
```

Profile one default CPU-only inference with Massif:

```bash
make valgrind_cpu
make ms_print_cpu
```

This profiles the default sample `CLASS=4 INDEX=9`, which keeps the Massif timeline, peak snapshot, and allocation call stack easier to inspect. Linux or WSL users may open `testbench/cpu/massif_out/massif.out.*_main` with `massif-visualizer`; arm64 macOS users should use the generated text report:

```text
testbench/cpu/massif_out/massif_output.txt
```

## DLA Flow

Run one DLA smoke test:

```bash
make test_dla
```

Run DLA simulation with per-layer statistics enabled for the report:

```bash
make test_dla_info
```

The DLA statistics CSV is written to:

```text
testbench/dla/dla_info.csv
```

The DLA simulator does not provide a 100-image full-test workflow because RTL simulation is much slower than the CPU-only host reference.

## Report Artifact Checklist

After `make homework`, fill `report.md` using these artifacts:

| Report item | Command | Artifact |
|-------------|---------|----------|
| HW 5.1 Relay graph before fusion | `make visuTVM` or `make visuTVM_origin` | `output/visu_VGG8_relay_ir.svg` |
| HW 5.1 Relay graph after fusion | `make visuTVM_pass` | `output/visu_VGG8_relay_ir_pass.svg` |
| HW 5.2 CPU-only accuracy | `make test_cpu_full` | `testbench/cpu/log/out.log` |
| HW 5.2 CPU-only Massif raw output | `make valgrind_cpu` | `testbench/cpu/massif_out/massif.out.*_main` |
| HW 5.2 CPU-only Massif text report | `make ms_print_cpu` | `testbench/cpu/massif_out/massif_output.txt` |
| HW 5.2 DLA statistics | `make test_dla_info` | `testbench/dla/dla_info.csv` |

## Clean Up

Remove generated code, testbench executables, logs, Massif outputs, and demo artifacts:

```bash
make clean
```

Remove the downloaded or generated dataset:

```bash
make clean_data
```

## Submission

Submit through GitHub Classroom and complete `report.md`. Program functionality is graded by the GitHub Classroom CI pipeline inside the provided Docker environment.

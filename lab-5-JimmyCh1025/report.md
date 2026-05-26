# Lab 5 - Homework Template

All commands below are intended to be run from the top-level `lab-5/` directory. You can run `make homework` once to generate the report artifacts, or use the individual commands listed in each section when you need to regenerate one artifact.

## HW 5.1 Codegen with TVM compiler (50%)

### Complete the python script (30%)
1. Explain how you implement `fuse_conv2d_bias_add_relu_max_pool2d()`, `fuse_dense_add_relu()`, and `fuse_dense_add()` in `fuse.py` (10%)
    - Describe how you identified the Relay/qnn ops used in your functions.
    - Record your investigation process, not a complete API reference.

    > Answer here

2. Explain how you implement `visit_call()` in `codegen.py` (20%)
    - Walk through your `visit_call()` implementation by tracing one specific composite
    - Record the decisions you made along the way, not a verbatim walk of the code.

    > Answer here

### Relay Graph using visuTVM (20%)
- Relay Graph before fuse PASS — locate one of your fuse chains (10%)
  - Pick any one of `fuse_conv2d_bias_add_relu_max_pool2d`, `fuse_dense_add_relu`, or `fuse_dense_add`, locate an instance of its chain on the pre-fuse Relay graph, and indicate the position with a cropped screenshot, an annotated overlay, or any visual method that makes the chain clear.
  - Generate the graph with: `make visuTVM` or `make visuTVM_origin`
  - Source file: `output/visu_VGG8_relay_ir.svg`

    Chosen function: `???`

    > ![Link to cropped/annotated image]()

- Relay Graph after fuse PASS (10%)
  - Generate the graph with: `make visuTVM_pass`
  - Source file: `output/visu_VGG8_relay_ir_pass.svg`

    > ![Link to *.svg file]()

## HW 5.2 Simulation and Performance Analysis (50%)
### Inference model with CPU-only (25%)
- Screenshot the `test_cpu_full` result matrix, including Accuracy (%)
  - Generate with: `make test_cpu_full`
  - Log file: `testbench/cpu/log/out.log`

    > ![Link to screenshot image]()

- Screenshot the massif-visualizer memory graph result for the single-image CPU run, or attach the `ms_print` text report, and record the peak memory usage in the report.
  - Generate with: `make valgrind_cpu`
  - Generate text report with: `make ms_print_cpu`
  - The Massif profile uses the default sample `CLASS=4 INDEX=9` so the memory timeline and allocation call stack are easier to inspect.
  - Open `testbench/cpu/massif_out/massif.out.*_main` with massif-visualizer.
  - Or read `testbench/cpu/massif_out/massif_output.txt` if massif-visualizer is unavailable.

    Memory Peak : ???

    > ![Link to screenshot image]()
    > [Link to massif_output.txt]()

### Inference model with DLA (25%)

- Fill the statistic data in to the sheet.
  - Generate with: `make test_dla_info`
  - CSV file: `testbench/dla/dla_info.csv`

    | Layer | Operation             | Cycles   | Time(ns) | Memory read | Memory write |
    |-------|-----------------------|----------|----------|-------------|--------------|
    |   1   |qconv2d_relu_maxpool   |          |          |             |              |
    |   2   |qconv2d_relu_maxpool   |          |          |             |              |
    |   3   |qconv2d_relu           |          |          |             |              |
    |   4   |qconv2d_relu           |          |          |             |              |
    |   5   |qconv2d_relu_maxpool   |          |          |             |              |

- Bar chart of **Cycles per Layer**
    ```mermaid
    %%{init: {
    "themeVariables": {
        "xyChart": {
        "plotColorPalette": "#1783b5"
        }
    }
    }}%%
    xychart-beta
        title "Cycles per Layer"
        x-axis ["layer 1", "layer 2", "layer 3", "layer 4", "layer 5"]
        y-axis "Cycles"
        bar [0, 0, 0, 0, 0]
    ```

- Bar chart of **Memory read per Layer**
    ```mermaid
    %%{init: {
    "themeVariables": {
        "xyChart": {
        "plotColorPalette": "#17b55e"
        }
    }
    }}%%
    xychart-beta
        title "Memory read per Layer"
        x-axis ["layer 1", "layer 2", "layer 3", "layer 4", "layer 5"]
        y-axis "Memory read (Bytes)"
        bar [0, 0, 0, 0, 0]
    ```

- Bar chart of **Memory write per Layer**
    ```mermaid
    %%{init: {
    "themeVariables": {
        "xyChart": {
        "plotColorPalette": "#e4b311"
        }
    }
    }}%%
    xychart-beta
        title "Memory write per Layer"
        x-axis ["layer 1", "layer 2", "layer 3", "layer 4", "layer 5"]
        y-axis "Memory write (Bytes)"
        bar [0, 0, 0, 0, 0]
    ```

## Reflection (bonus 5%)
(more reflection, more credits)

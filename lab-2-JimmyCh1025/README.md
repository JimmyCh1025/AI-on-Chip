[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/S3MDplrq)
# Lab 2: Workload Analysis and Performance Modeling

This document explains how to set up the environment, manage model weights, run workload profiling, execute Design Space Exploration (DSE), and submit your assignment for **Lab 2**.

---

## 1. Environment Setup

### Step 1: Clone the Repository

Accept the assignment from **GitHub Classroom**, then clone the repository into your Docker environment.

```bash
git clone <your-repo-url>
cd <repo-name>
```

---

### Step 2: Download Model Weights

Download the **weights** folder from Moodle and place it in the project root/src.

Expected directory structure:

```
$PROJECT_ROOT/
├── src/
│   ├── weights/
│   │   ├── vgg8.pt
│   │   └── vgg8-power2.pt
│   └── ...
└── ...
```

---

## 2. Workload Analysis (`profiling.py`)

Use `profiling.py` to analyze computation breakdown for each operator.

### View Help

```bash
python3 profiling.py -h
```

### Profile Full-Precision Model (FP32)

```bash
python3 profiling.py ./weights/vgg8.pt
```

### Profile Quantized Model (INT8)

Specify quantization backend (e.g., `power2`):

```bash
python3 profiling.py ./weights/vgg8-power2.pt -b power2
```

---

## 3. Main Execution (`main.py`)

`main.py` integrates:

* Network Parser
* Analytical Model
* Design Space Exploration (DSE)

---

### View Help

```bash
python3 main.py -h
```

### Basic Usage (PyTorch Model)

```bash
python3 main.py ./weights/vgg8-power2.pt
```

### Advanced Usage

```bash
python3 main.py -f torch -o log/$(date -I) --plot --mode hardware weights/vgg8-power2.pt
```

---

## 4. Submission Guidelines

### Deadline

**April 06, 2026 (Monday) 23:59:59**
Late submissions will **NOT** be accepted.

---

### Required Submission Structure

```
$PROJECT_ROOT
├── README.md
├── report.md                     # Lab report
├── images/
│   └── image_for_report
├── src/
│   ├── __init__.py
│   ├── analytical_model/
│   │   ├── __init__.py
│   │   ├── eyeriss.py            
│   │   └── mapper.py             
│   ├── layer_info.py
│   ├── lib/
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── lenet.py
│   │   │   ├── mlp.py
│   │   │   ├── qconfig.py
│   │   │   └── vgg.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── dataset.py
│   │       └── utils.py
│   ├── main.py                   
│   ├── network_parser/
│   │   ├── __init__.py
│   │   ├── network_parser.py     
│   │   └── torch2onnx.py
│   ├── onnx_inference.py
│   ├── profiling.py            
│   └── roofline.py
│   
│       
├── test/                         # Grading tests (DO NOT MODIFY)
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_analytical_model.py
│   ├── test_dse.py
│   ├── test_network_parser.py
│   └── utils.py
└── [Other implemented files (optional)]

```

---

## 5. How to Submit via GitHub Classroom
For more detailed instructions on using Git, please refer to Lab 0.
Follow these steps to submit your lab:

### Step 1: Check Your Changes

Verify modified files:

```bash
git status
```

---

### Step 2: Add Files to Commit

```bash
git add .
```

---

### Step 3: Commit Your Work

```bash
git commit -m "fix: GLB usage formula"
```

You may commit multiple times before the deadline.

---

### Step 4: Push to GitHub Classroom

```bash
git push origin main
```

---

### Step 5: Verify Submission

1. Go to your **GitHub Classroom repository page**.
2. Confirm latest commit is pushed.
3. Ensure:
   * Required files exist
   * No changes in `test/` and `.github/`
4. Wait a moment for the autograding results.
---

### Notes

* You can push multiple times; **the latest commit before the deadline is graded**.
* If you forget to push, your work is **not submitted**.
* Always verify on GitHub after pushing.

---

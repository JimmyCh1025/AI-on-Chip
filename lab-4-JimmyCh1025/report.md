# Lab 4 - Homework Template

> [!Caution] AI Usage Guidelines
> You may use AI assistants to help organize and polish your report, but you are responsible for **verifying** that all technical claims are **correct** and supported by **your own implementation and measurements**.
> Reports containing incorrect statements, unsupported conclusions, or padded content will receive **reduced credit**.

## 1. Performance Summary Table (5 Points)

> [!Note] Grading Criteria
> To receive full credit, you must fill in the performance summary table completely.

| Case  | CPU original (cycles) | CPU improve (cycles) | CPU Cycle Reduction (%) | DLA (cycles) | CPU improve vs. DLA Speedup |
|-------|----------------------|---------------------|-------------------------|--------------|-----------------------------|
| case0 (`qconv2d_relu`) | 957,120 | | | | |
| case1 (`qconv2d_relu_maxpool`) | 72,265,534 | | | | |
| case2 (`qconv2d_relu_maxpool`) | 1,004,896,044 | | | | |
| case3 (`linear`) | 23,503,198 | | | N/A | N/A |
| case4 (`linear_relu`) | 23,505,126 | | | N/A | N/A |

> **CPU Cycle Reduction** = $\frac{Cycle_{original} - Cycle_{improve}}{Cycle_{original}} \times 100\%$
> 
> **DLA Speedup** = $\frac{CPU_{improve}\ cycles}{DLA\ cycles}$

---

## 2. Optimization Methods and Scratch Buffer Design (20 Points)

> [!Note] Grading Criteria
> This section evaluates whether you can clearly explain what you changed in the `improve` version and why those changes are reasonable.
> To receive full credit, you must:
> - Describe at least **2 optimization methods** clearly
> - Explain the purpose of each method
> - Connect each method to the workload or memory-access pattern it is meant to improve
> - Justify the `SCRATCH_SIZE` settings for all required cases

### 2.1 Optimization Methods (10 Points, 5 Points Each)

> Describe at least **2 optimization methods** implemented in your `improve` version. Focus on what each method is trying to improve and how you implemented it.
> 
> If multiple optimizations are combined together, you may describe their impact **cumulatively**. You **do not** need to isolate the exact performance contribution of each method.

> [!Tip]
> For each method, make sure your description clearly explains the following points:
> - What problem in the original code are you trying to improve?
> - What did you change in the `improve` version?
> - Why should this change help performance?
> - Which cases benefit the most from this method?

#### Method 1: [Method Name]

- **What problem does this method target?**
- **What did you change?**
- **Why should this help?**
- **Which cases does it mainly affect?**


```cpp
// Paste the optimized code section here
```

#### Method 2: [Method Name]

- **What problem does this method target?**
- **What did you change?**
- **Why should this help?**
- **Which cases does it mainly affect?**

```cpp
// Paste the optimized code section here
```

*(Add more methods as needed...)*

---

### 2.2 Scratch Buffer Size Design (10 Points, 2 Points Each)

> For each case, report the minimum `SCRATCH_SIZE`, the final `SCRATCH_SIZE`, and whether size tuning changed the performance.

> [!Tip]
> In the tuning result, briefly state whether the final value is the same as the minimum, or whether a different size gave better performance.

#### case0 (`qconv2d_relu`)
- **Minimum `SCRATCH_SIZE`:**
- **Final `SCRATCH_SIZE`:**
- **Tuning result:**

#### case1 (`qconv2d_relu_maxpool`)
- **Minimum `SCRATCH_SIZE`:**
- **Final `SCRATCH_SIZE`:**
- **Tuning result:**

#### case2 (`qconv2d_relu_maxpool`)
- **Minimum `SCRATCH_SIZE`:**
- **Final `SCRATCH_SIZE`:**
- **Tuning result:**

#### case3 (`linear`)
- **Minimum `SCRATCH_SIZE`:**
- **Final `SCRATCH_SIZE`:**
- **Tuning result:**

#### case4 (`linear_relu`)
- **Minimum `SCRATCH_SIZE`:**
- **Final `SCRATCH_SIZE`:**
- **Tuning result:**

---

## 3. CPU Profiling and Bottleneck Analysis (20 Points)

> This section evaluates whether you can use measured statistics to **explain** performance behavior.

> [!Note] Grading Criteria
> To receive full credit, you must:
> - Explain your miss penalty estimation design clearly
> - Fill in the cache statistics table completely
> - Use the statistics to support your bottleneck analysis with concrete evidence

### 3.1 Miss Penalty Estimation (5 Points)

> Explain your miss penalty estimation method by answering the following:
> 1. What event did you use as the start of an **L1D miss**? What event did you use as the end?
> 2. What event did you use as the start of an **L2 miss**? What event did you use as the end?
> 3. Why are these values **only estimations**, rather than exact CPU stall cycles?
> 4. If an L1D miss also triggers an L2 miss, can the estimated L1D and L2 penalties overlap? If yes, explain how this affects the interpretation of your numbers.

> [!Tip]
> Focus on how you define the measurement window and how you interpret the result.
> Your explanation should make clear why the measured penalty is useful and what its limitations are.

#### Code Section
```cpp
// Paste your miss penalty estimation code here
```

#### Explanation
[Your Answer Here]

---

### 3.2 Cache Statistics Comparison (5 Points)

> You have already implemented your own miss penalty estimation in the HAL.
> In this section, fill in the measured cache and memory statistics into the table below.
> These statistics will be used as the main evidence for your bottleneck analysis in Section 3.3.

| Case  | Version | L1D$ Hit Rate | L2$ Hit Rate | L1D$ Penalty (cycles/miss) | L2$ Penalty (cycles/miss) | DRAM Read (B) | DRAM Write (B) |
|-------|---------|---------------|--------------|----------------------------|---------------------------|---------------|----------------|
| case0 | original | | | | | | |
| case0 | improve  | | | | | | |
| case1 | original | | | | | | |
| case1 | improve  | | | | | | |
| case2 | original | | | | | | |
| case2 | improve  | | | | | | |
| case3 | original | | | | | | |
| case3 | improve  | | | | | | |
| case4 | original | | | | | | |
| case4 | improve  | | | | | | |

> [!Note] Collecting Kernel Statistics
> You should use `make run_cpu_caseN` to run the original kernel and collect the statistics for the `original` rows in the table, where `N` is the case number.
>
> Note that `case2` with the original kernel may take around **20–25 minutes**, depending on your machine performance. Please be patient and wait for the simulation to complete.

---

### 3.3 Bottleneck and Cache Analysis (10 Points, 5 Points Each)

> Analyze the following **2 representative cases** in detail:
> - **case2** (`qconv2d_relu_maxpool`)
> - **case4** (`linear_relu`)
>
> In this section, focus on the **measured behavior** of the original and improved versions.
> You should use the statistics from Section 3.2 to explain where the bottleneck is and why performance improved.

#### case2 (`qconv2d_relu_maxpool`)

- **Main bottleneck in the original version:**
- **Key evidence from statistics:**
- **How the improved version changed the behavior:**
- **Why the cycle count decreased:**


#### case4 (`linear_relu`)

- **Main bottleneck in the original version:**
- **Key evidence from statistics:**
- **How the improved version changed the behavior:**
- **Why the cycle count decreased:**


---

## 4. Reflection and Feedback (5 Points)
> [!Note] Grading Criteria
> To receive full credit, your response should include both:
> - Reflection on your implementation and optimization results
> - Constructive feedback on the lab

> In one short section, discuss:
> - The most effective optimization method you implemented
> - The least effective or most difficult part
> - Any unexpected behavior of the improved version
> - Your feedback on the lab

[Your Answer Here]

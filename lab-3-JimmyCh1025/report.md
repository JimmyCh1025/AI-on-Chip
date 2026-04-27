# Lab 3 - Homework Template

## Lab Design Description (60%)
Scores will be assigned based on the level of detail and the logical soundness of your description.
Each section should ideally include a block diagram to explain the module.
> [!Warning]
> Points will be deducted for hand-drawn block diagrams (including those drawn on an iPad).

### Explain how you implement PE (20%)
> Including the FSM of your PE and how you handle the computation of the zero point of dequantization.
### Explain how you implement PE array (30%)
> Including network (GIN/GON/LN) and multicast controller (MC)

### Explain how you implement PPU (10%)
> Including how you handle the computation of the zero point of requantization

### Result

| Component | Pass (Y/N) |
|:---------:|:----------:|
|    PE     |            |
| PE array  |            |
|    PPU    |            |


## Question (40%)

### Question 1 (20%)

Explain how data reuse is achieved in the design presented in the Eyeriss paper.

### Question 2 (10%)

Compute a `16×16` Conv2D operation, given the following configuration
- Kernel size: `3×3`
- Stride: `1`
- Padding: `1`
- Global Buffer (GLB) size: `128 KB`
- Mapping parameters:
    - `p = 4`
    - `q = 4`
    - `r = 1`
    - `t = 2`
    - `e = 8`

Determine the value of the mapping parameter `m`

> [!Warning]
> Please include the calculation process.

### Question 3 (10%)

For the test cases where $e = 4$, configurations $(r, t) = (1, 4)$ and $(2, 2)$ are present, but $(4, 1)$ is not. Could you explain the reason behind this omission?

## Feedback (Bonus 10%)
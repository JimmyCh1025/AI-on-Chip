[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/oom3nb0w)
## Submission Guidelines

### Deadline

- **2026/5/4 23:59:59**
- **Late submissions will not be accepted**

### File Hierarchy

``` 
Lab3
├── .gitignore
├── .github/               # Do not modify auto-grading script
│   └── workflows/
│       └── classroom.yml
├── Makefile               # Do not modify Makefile
├── report.md
├── art/
├── include/
├── wave/        
├── images/                # Place images used in report.md here
├── testbench/             # Do not modify testbench and test data
└── src/
    ├── PE_array/
    │   ├── GIN/
    │   │   ├── GIN.sv
    │   │   ├── GIN_MulticastController.sv
    │   │   └── GIN_bus.sv
    │   ├── GON/
    │   │   ├── GON.sv
    │   │   ├── GON_MulticastController.sv
    │   │   └── GON_bus.sv
    │   ├── PE.sv
    │   └── PE_array.sv
    └── PPU/
        ├── PPU.sv
        ├── PostQuant.sv
        ├── Maxpool_Qint8.sv
        └── ReLU_Qint8.sv
```

### Important Notes

- Your work will be graded through **GitHub Classroom**, running inside the **provided Docker environment**.
- Program Functionality is determined **solely** by the **output produced by the GitHub Classroom CI pipeline**.
- `Makefile` and files under `.github/`, `testbench/` are relevant to the auto-grading mechanism and are marked as **protected**; therefore, <font color="#f00">**modifying these files is strictly prohibited.**</font>
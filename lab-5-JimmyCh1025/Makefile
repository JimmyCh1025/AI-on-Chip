# Lab 5 top-level workflow

MODEL          := ./model/alexnetbn_v2-power2.onnx
OUTPUT_DIR     := ./output
RELAY_DIR      := $(OUTPUT_DIR)/relay
BIN_DIR        := $(OUTPUT_DIR)/bin
BUILD_SCRIPT   := ./Python/build_model.py
VISUTVM_SCRIPT := Python/VisuTVM/main.py
TESTBENCH_DIR  := ./testbench
LAB_DIR        := ./lab
DATASET_DIR    := ./data
CACHE_DIR      := ./Python/utils/__pycache__
CPU_MASSIF_DIR := $(TESTBENCH_DIR)/cpu/massif_out
CPU_MASSIF_TXT := $(CPU_MASSIF_DIR)/massif_output.txt

PYTHON  ?= python3
MS_PRINT ?= ms_print
QUIET   ?= @
ifeq ($(QUIET),0)
  QUIET :=
endif

# Terminal colors
C_RESET  := \033[0m
C_GREEN  := \033[32m
C_YELLOW := \033[33m
C_RED    := \033[31m
C_BLUE   := \033[34m

MSG_INFO     = printf "$(C_BLUE)[INFO]$(C_RESET) %s\n"
MSG_OK       = printf "$(C_GREEN)[OK]$(C_RESET) %s\n"
MSG_WARN     = printf "$(C_YELLOW)[WARNING]$(C_RESET) %s\n"
MSG_ERROR    = printf "$(C_RED)[ERROR]$(C_RESET) %s\n"
MSG_ARTIFACT = printf "  %s: $(C_GREEN)%s$(C_RESET)\n"

.DEFAULT_GOAL := help

.PHONY: help homework all build_model visuTVM visuTVM_origin visuTVM_pass \
        massif_demo ms_print_demo \
        test_cpu test_dla test_cpu_full valgrind_cpu ms_print_cpu test_dla_info \
        clean clean_data compdb format

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Main targets:"
	@echo "  homework           Build every artifact needed by report.md"
	@echo "  all                Build model, run CPU/DLA smoke tests, draw Relay graphs"
	@echo "  build_model        Generate TVM libraries, Relay dumps, and input.bin"
	@echo "  visuTVM            Generate before-fusion graph, then try after-fusion graph"
	@echo "  visuTVM_origin     Generate before-fusion Relay graph SVG"
	@echo "  visuTVM_pass       Generate after-fusion Relay graph SVG (strict)"
	@echo ""
	@echo "Learning exercise:"
	@echo "  massif_demo        Profile the simple Fibonacci demo with Massif"
	@echo "  ms_print_demo      Convert the Fibonacci demo Massif output to text"
	@echo ""
	@echo "Report artifacts:"
	@echo "  test_cpu_full      Run CPU-only inference on all 100 test images"
	@echo "  valgrind_cpu       Run one CPU-only inference under Valgrind Massif"
	@echo "  ms_print_cpu       Convert CPU Massif output to a text report"
	@echo "  test_dla_info      Run DLA inference and dump per-layer CSV statistics"
	@echo ""
	@echo "Smoke tests:"
	@echo "  test_cpu           Run one CPU-only inference"
	@echo "  test_dla           Run one DLA inference"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean              Remove build/test artifacts"
	@echo "  clean_data         Remove downloaded/generated dataset"
	@echo "  compdb             Generate compile_commands.json using bear"
	@echo "  format             Format C/C++ and Python sources"
	@echo ""
	@echo "Options:"
	@echo "  QUIET=0            Show commands while running"

all: build_model test_cpu test_dla visuTVM

homework: build_model
	$(QUIET)$(MAKE) visuTVM_origin
	$(QUIET)$(MAKE) visuTVM_pass
	$(QUIET)$(MAKE) test_cpu_full
	$(QUIET)$(MAKE) valgrind_cpu
	$(QUIET)$(MAKE) ms_print_cpu
	$(QUIET)$(MAKE) test_dla_info
	$(QUIET)$(MSG_OK) "Homework artifacts are ready:"
	$(QUIET)$(MSG_ARTIFACT) "CPU full-test log      " "$(TESTBENCH_DIR)/cpu/log/out.log"
	$(QUIET)$(MSG_ARTIFACT) "CPU Massif output      " "$(CPU_MASSIF_DIR)/massif.out.*_main"
	$(QUIET)$(MSG_ARTIFACT) "CPU Massif text report " "$(CPU_MASSIF_TXT)"
	$(QUIET)$(MSG_ARTIFACT) "DLA per-layer CSV      " "$(TESTBENCH_DIR)/dla/dla_info.csv"
	$(QUIET)$(MSG_ARTIFACT) "Relay visualization SVG" "$(OUTPUT_DIR)/visu_VGG8_relay_ir*.svg"

BUILD_STAMP := $(OUTPUT_DIR)/.build_stamp

build_model: $(BUILD_STAMP)

$(BUILD_STAMP): $(MODEL) $(BUILD_SCRIPT)
	$(QUIET)$(MSG_INFO) "Building TVM model and dataset..."
	$(QUIET)mkdir -p $(OUTPUT_DIR) $(RELAY_DIR) $(BIN_DIR)
	$(QUIET)$(PYTHON) $(BUILD_SCRIPT) -m $(MODEL) -o $(OUTPUT_DIR) -l $(RELAY_DIR) -d $(DATASET_DIR)
	$(QUIET)mv $(OUTPUT_DIR)/input.bin $(BIN_DIR)/ 2>/dev/null || true
	$(QUIET)touch $@
	$(QUIET)$(MSG_OK) "Build artifacts are ready:"
	$(QUIET)$(MSG_ARTIFACT) "Output directory" "$(OUTPUT_DIR)"
	$(QUIET)$(MSG_ARTIFACT) "Relay directory " "$(RELAY_DIR)"
	$(QUIET)$(MSG_ARTIFACT) "Binary directory" "$(BIN_DIR)"

visuTVM: visuTVM_origin
	$(QUIET)$(MSG_INFO) "Trying to draw after-fusion Relay graph..."
	$(QUIET)if $(MAKE) --no-print-directory visuTVM_pass >/dev/null 2>&1; then \
	  $(MSG_OK) "After-fusion Relay graph is ready:"; \
	  $(MSG_ARTIFACT) "Relay graph SVG" "$(OUTPUT_DIR)/visu_VGG8_relay_ir_pass.svg"; \
	else \
	  $(MSG_WARN) "Fusion is not finished; keeping the before-fusion graph:"; \
	  $(MSG_ARTIFACT) "Relay graph SVG" "$(OUTPUT_DIR)/visu_VGG8_relay_ir.svg"; \
	  $(MSG_WARN) "After-fusion graph was not generated. Complete the fusion pattern and run 'make visuTVM_pass' to see the full error."; \
	fi

visuTVM_origin:
	$(QUIET)$(MSG_INFO) "Drawing before-fusion Relay graph..."
	$(QUIET)mkdir -p $(OUTPUT_DIR) $(RELAY_DIR)
	$(QUIET)$(PYTHON) $(BUILD_SCRIPT) -m $(MODEL) -o $(OUTPUT_DIR) -l $(RELAY_DIR) -d $(DATASET_DIR) --dump-origin-only
	$(QUIET)$(PYTHON) $(VISUTVM_SCRIPT) \
	  -ri $(RELAY_DIR)/relay_mc_origin_model.txt \
	  -sn VGG8 \
	  -wi
	$(QUIET)$(MSG_OK) "Before-fusion Relay graph is ready:"
	$(QUIET)$(MSG_ARTIFACT) "Relay graph SVG" "$(OUTPUT_DIR)/visu_VGG8_relay_ir.svg"

visuTVM_pass:
	$(QUIET)$(MSG_INFO) "Drawing after-fusion Relay graph..."
	$(QUIET)mkdir -p $(OUTPUT_DIR) $(RELAY_DIR)
	$(QUIET)$(PYTHON) $(BUILD_SCRIPT) -m $(MODEL) -o $(OUTPUT_DIR) -l $(RELAY_DIR) -d $(DATASET_DIR) --dump-relay-only
	$(QUIET)$(PYTHON) $(VISUTVM_SCRIPT) \
	  -ri $(RELAY_DIR)/relay_mc_MergeComposite_model.txt \
	  -pn MergeComposite \
	  -sn VGG8 \
	  -wi
	$(QUIET)$(MSG_OK) "After-fusion Relay graph is ready:"
	$(QUIET)$(MSG_ARTIFACT) "Relay graph SVG" "$(OUTPUT_DIR)/visu_VGG8_relay_ir_pass.svg"

massif_demo:
	$(QUIET)$(MAKE) -C $(LAB_DIR) massif
	$(QUIET)$(MSG_OK) "Fibonacci demo Massif output is ready:"
	$(QUIET)$(MSG_ARTIFACT) "Massif output" "$(LAB_DIR)/massif_out/massif.out.massif_test"

ms_print_demo:
	$(QUIET)$(MAKE) -C $(LAB_DIR) ms_print
	$(QUIET)$(MSG_OK) "Fibonacci demo ms_print report is ready:"
	$(QUIET)$(MSG_ARTIFACT) "ms_print report" "$(LAB_DIR)/massif_out/massif_output.txt"

test_cpu: $(BUILD_STAMP)
	$(QUIET)$(MAKE) -C $(TESTBENCH_DIR)/cpu clean
	$(QUIET)$(MAKE) -C $(TESTBENCH_DIR)/cpu all
	$(QUIET)$(MAKE) -C $(TESTBENCH_DIR)/cpu test

test_dla: $(BUILD_STAMP)
	$(QUIET)$(MAKE) -C $(TESTBENCH_DIR)/dla clean
	$(QUIET)$(MAKE) -C $(TESTBENCH_DIR)/dla all
	$(QUIET)$(MAKE) -C $(TESTBENCH_DIR)/dla test

test_cpu_full: $(BUILD_STAMP)
	$(QUIET)$(MAKE) -C $(TESTBENCH_DIR)/cpu clean
	$(QUIET)$(MAKE) -C $(TESTBENCH_DIR)/cpu all
	$(QUIET)$(MAKE) -C $(TESTBENCH_DIR)/cpu test_full
	$(QUIET)$(MSG_OK) "CPU full-test log is ready:"
	$(QUIET)$(MSG_ARTIFACT) "CPU full-test log" "$(TESTBENCH_DIR)/cpu/log/out.log"

valgrind_cpu: $(BUILD_STAMP)
	$(QUIET)$(MAKE) -C $(TESTBENCH_DIR)/cpu all
	$(QUIET)$(MAKE) -C $(TESTBENCH_DIR)/cpu valgrind
	$(QUIET)$(MSG_OK) "CPU Massif output is ready:"
	$(QUIET)$(MSG_ARTIFACT) "CPU Massif output" "$(CPU_MASSIF_DIR)/massif.out.*_main"

ms_print_cpu: $(BUILD_STAMP)
	$(QUIET)set -e; \
	  latest=$$(ls -t $(CPU_MASSIF_DIR)/massif.out.*_main 2>/dev/null | head -n 1 || true); \
	  if [ -z "$$latest" ]; then \
	    $(MSG_WARN) "No CPU Massif output found; running valgrind_cpu first..."; \
	    $(MAKE) valgrind_cpu; \
	    latest=$$(ls -t $(CPU_MASSIF_DIR)/massif.out.*_main 2>/dev/null | head -n 1 || true); \
	  fi; \
	  if [ -z "$$latest" ]; then \
	    $(MSG_ERROR) "No CPU Massif output found in $(CPU_MASSIF_DIR)" >&2; \
	    exit 1; \
	  fi; \
	  $(MSG_INFO) "Generating CPU Massif text report from $$latest"; \
	  $(MS_PRINT) "$$latest" > $(CPU_MASSIF_TXT); \
	  $(MSG_OK) "CPU Massif text report is ready:"; \
	  $(MSG_ARTIFACT) "CPU Massif text report" "$(CPU_MASSIF_TXT)"

test_dla_info: $(BUILD_STAMP)
	$(QUIET)$(MAKE) -C $(TESTBENCH_DIR)/dla clean
	$(QUIET)$(MAKE) -C $(TESTBENCH_DIR)/dla all DLA_INFO=1
	$(QUIET)$(MAKE) -C $(TESTBENCH_DIR)/dla test DLA_INFO=1
	$(QUIET)$(MSG_OK) "DLA per-layer CSV is ready:"
	$(QUIET)$(MSG_ARTIFACT) "DLA per-layer CSV" "$(TESTBENCH_DIR)/dla/dla_info.csv"

clean:
	$(QUIET)$(MAKE) -C $(TESTBENCH_DIR)/dla clean
	$(QUIET)$(MAKE) -C $(TESTBENCH_DIR)/cpu clean
	$(QUIET)$(MAKE) -C $(LAB_DIR) clean
	$(QUIET)rm -rf $(OUTPUT_DIR) $(CACHE_DIR) compile_commands.json .ruff_cache
	$(QUIET)$(MSG_OK) "Cleaned build/test artifacts"

clean_data:
	$(QUIET)rm -rf $(DATASET_DIR)
	$(QUIET)$(MSG_OK) "Removed dataset directory:"
	$(QUIET)$(MSG_ARTIFACT) "Dataset directory" "$(DATASET_DIR)"

compdb: clean
	$(QUIET)bear -- $(MAKE) all
	$(QUIET)$(MSG_OK) "Compilation database is ready:"
	$(QUIET)$(MSG_ARTIFACT) "Compilation database" "compile_commands.json"

EXCLUDE_DIRS  := output data model __pycache__ .git obj massif_out log nWave_log hardware .venv
FIND_EXCLUDES := $(foreach dir,$(EXCLUDE_DIRS),-name "$(dir)" -prune -o)

format:
	$(QUIET)find . $(FIND_EXCLUDES) -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) -print0 | xargs -0 -r clang-format -i
	$(QUIET)find . $(FIND_EXCLUDES) -type f -name "*.py" -print0 | xargs -0 -r $(PYTHON) -m ruff format --line-length 120
	$(QUIET)$(MSG_OK) "Formatted C/C++ and Python sources"
# test/testbench/cpu/Makefile
# Build and run NutShell CPU Runtime Testbench

RESET  := \033[0m
GREY   := \033[0;37m
WHITE  := \033[1;37m
GREEN  := \033[0;32m
RED    := \033[0;31m
CYAN   := \033[0;36m

define msg_grey
	@printf "$(GREY)$(1)$(RESET)\n"
endef
define msg_green
	@printf "$(GREEN)$(1)$(RESET)\n\n"
endef
define msg_red
	@printf "$(RED)$(1)$(RESET)\n"
endef
define msg_white
	@printf "$(WHITE)$(1)$(RESET)\n"
endef
define msg_cyan
	@printf "$(CYAN)$(1)$(RESET)\n"
endef

LAB4_ROOT    := $(abspath ../../..)
TB_DIR       := $(abspath .)
RTL_DIR      := $(LAB4_ROOT)/src/hardware/cpu/rtl
HAL_INC_DIR  := $(LAB4_ROOT)/include/hal
HAL_SRC_DIR  := $(LAB4_ROOT)/src/hal
HOST_SRC_DIR := $(LAB4_ROOT)/src/host
CPU_ORIG_DIR := $(LAB4_ROOT)/src/runtime/cpu/original
BUILD_DIR    := $(TB_DIR)/build
RUNTIME_DIR  := $(LAB4_ROOT)/src/runtime/cpu
TB_ARGS      ?=

VERILATOR   = verilator
VL_FLAGS    = --cc --trace-fst --build -j 0 -exe
VL_FLAGS   += -I$(RTL_DIR)
VL_FLAGS   += -I$(HAL_INC_DIR)
VL_FLAGS   += -CFLAGS "-I$(HAL_INC_DIR) -I$(TB_DIR) -I$(LAB4_ROOT)/test/cases \
                        -I$(HOST_SRC_DIR) -I$(CPU_ORIG_DIR)"
VL_FLAGS   += --Mdir $(BUILD_DIR)/obj_dir
VL_FLAGS   += -o $(TB_EXE)
VL_FLAGS   += --timescale 1ns/1ps
VL_FLAGS   += --relative-includes
VL_FLAGS   += -Wno-SYMRSVDWORD
VL_FLAGS   += -Wno-DECLFILENAME
VL_FLAGS   += -Wno-WIDTHTRUNC
VL_FLAGS   += -Wno-UNOPTFLAT
VL_FLAGS   += -Wno-CASEINCOMPLETE
VL_FLAGS   += +define+SYNTHESIS

RTL_SV_TOP  = $(RTL_DIR)/NutShell.sv
RTL_SV_REST = $(shell find $(RTL_DIR) -name "*.sv" \
                ! -name "NutShell.sv"    \
                ! -name "NutShellSim.sv" \
                ! -name "SimTop.sv"      \
                ! -name "SimMMIO.sv" 2>/dev/null)
RTL_V       = $(shell find $(RTL_DIR) -name "*.v" 2>/dev/null)
ASSERT_V    = $(TB_DIR)/assert.v

TB_EXE = $(BUILD_DIR)/tb_cpu

CASES   = 0 1 2
CASES_FALLBACK = 4 5
CASES_FALLBACK_NAMES = case_cpu_fallback_linear case_cpu_fallback_linear_relu
VARIANTS = original improve

CASE_ELF = $(RUNTIME_DIR)/case$(1)_$(2).elf
ROOT_BUILD_DIR := $(LAB4_ROOT)/build

.PHONY: all tb run_all_cases clean
.PHONY: $(foreach c,$(CASES),$(foreach v,$(VARIANTS),run_case$(c)_$(v)))
.PHONY: $(foreach c,$(CASES),case$(c)_trace case$(c)_trace_improve)
.PHONY: $(foreach n,$(CASES_FALLBACK_NAMES),$(foreach v,$(VARIANTS),run_$(n)_$(v)))
.PHONY: $(foreach n,$(CASES_FALLBACK_NAMES),$(n)_trace $(n)_trace_improve)

all: tb

tb: $(TB_EXE)

$(TB_EXE): $(RTL_SV_TOP) $(RTL_SV_REST) $(RTL_V) \
           $(TB_DIR)/tb.cpp $(TB_DIR)/elf_loader.h $(TB_DIR)/sram_model.h \
           $(HAL_SRC_DIR)/cpu_hal.cpp $(HAL_INC_DIR)/cpu_hal.hpp \
           $(HAL_INC_DIR)/hal.hpp \
           $(HOST_SRC_DIR)/quantize.c $(HOST_SRC_DIR)/quantize.h \
		   $(CPU_ORIG_DIR)/kernel_cpu.c $(CPU_ORIG_DIR)/kernel_cpu.h
	@mkdir -p $(BUILD_DIR)/obj_dir
	$(call msg_grey,[TB/CPU] Verilating NutShell + CpuHAL...)
	$(VERILATOR) $(VL_FLAGS) \
	    --top-module NutShell \
	    --exe $(TB_DIR)/tb.cpp $(HAL_SRC_DIR)/cpu_hal.cpp \
	        $(HOST_SRC_DIR)/quantize.c $(CPU_ORIG_DIR)/kernel_cpu.c \
	    $(ASSERT_V) $(RTL_SV_TOP) $(RTL_SV_REST) $(RTL_V)
	$(call msg_green,[TB/CPU] tb_cpu ready at $(TB_EXE))

define RUN_CASE_TARGET
.PHONY: run_case$(1)_$(2)
run_case$(1)_$(2): $(TB_EXE)
	@$(MAKE) -C $(RUNTIME_DIR) case$(1)_$(2).elf
	@mkdir -p $(ROOT_BUILD_DIR)
	$(call msg_cyan,[TB/CPU] Running case$(1) $(2) ...)
	$(TB_EXE) --case $(1) $(if $(filter improve,$(2)),--improve) \
	    --trace-file $(ROOT_BUILD_DIR)/cpu_trace_case$(1).fst $(TB_ARGS)
endef

$(foreach c,$(CASES),$(foreach v,$(VARIANTS),$(eval $(call RUN_CASE_TARGET,$(c),$(v)))))

# case_cpu_fallback run targets
# $(1) = case number, $(2) = case name, $(3) = variant
define RUN_FALLBACK_TARGET
.PHONY: run_$(2)_$(3)
run_$(2)_$(3): $(TB_EXE)
	@$(MAKE) -C $(RUNTIME_DIR) $(2)_$(3)
	@mkdir -p $(ROOT_BUILD_DIR)
	$(call msg_cyan,[TB/CPU] Running $(2) $(3) ...)
	$(TB_EXE) --case $(1) $(if $(filter improve,$(3)),--improve) \
	    --trace-file $(ROOT_BUILD_DIR)/cpu_trace_$(2).fst $(TB_ARGS)
endef

$(eval $(call RUN_FALLBACK_TARGET,3,case_cpu_fallback_linear,original))
$(eval $(call RUN_FALLBACK_TARGET,3,case_cpu_fallback_linear,improve))
$(eval $(call RUN_FALLBACK_TARGET,4,case_cpu_fallback_linear_relu,original))
$(eval $(call RUN_FALLBACK_TARGET,4,case_cpu_fallback_linear_relu,improve))

define TRACE_CASE_TARGET
.PHONY: case$(1)_trace case$(1)_trace_improve
case$(1)_trace: $(TB_EXE) $(call CASE_ELF,$(1),original)
	$(call msg_cyan,[TB/CPU] Running case$(1) original with trace ...)
	$(TB_EXE) --case $(1) --trace
case$(1)_trace_improve: $(TB_EXE) $(call CASE_ELF,$(1),improve)
	$(call msg_cyan,[TB/CPU] Running case$(1) improve with trace ...)
	$(TB_EXE) --case $(1) --improve --trace
endef

$(foreach c,$(CASES),$(eval $(call TRACE_CASE_TARGET,$(c))))

# case_cpu_fallback trace targets
# $(1) = case number (4..7), $(2) = case name
define TRACE_FALLBACK_TARGET
.PHONY: $(2)_trace $(2)_trace_improve
$(2)_trace: $(TB_EXE) $(RUNTIME_DIR)/$(2)_original.elf
	$(call msg_cyan,[TB/CPU] Running $(2) original with trace ...)
	$(TB_EXE) --case $(1) --trace
$(2)_trace_improve: $(TB_EXE) $(RUNTIME_DIR)/$(2)_improve.elf
	$(call msg_cyan,[TB/CPU] Running $(2) improve with trace ...)
	$(TB_EXE) --case $(1) --improve --trace
endef

$(eval $(call TRACE_FALLBACK_TARGET,3,case_cpu_fallback_linear))
$(eval $(call TRACE_FALLBACK_TARGET,4,case_cpu_fallback_linear_relu))

run_all_cases: $(TB_EXE)
	@for c in $(CASES); do \
	    for v in $(VARIANTS); do \
	        printf "\n"; \
	        printf "\033[0;36m[TB/CPU] === case$${c} $${v} ===\033[0m\n\n"; \
	        if [ "$$v" = "improve" ]; then \
	            $(TB_EXE) --case $${c} --improve $(TB_ARGS); \
	        else \
	            $(TB_EXE) --case $${c} $(TB_ARGS); \
	        fi; \
	    done; \
	done

clean:
	$(call msg_grey,[TB/CPU] Cleaning...)
	@rm -rf $(BUILD_DIR)
	@rm -f $(ROOT_BUILD_DIR)/cpu_trace_*.fst

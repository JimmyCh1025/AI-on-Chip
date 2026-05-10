# test/testbench/dla/Makefile
# Build and run DLA testbench for all test cases (case0~2)

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

LAB4_ROOT := $(abspath ../../..)
TB_DIR    := $(abspath .)

CXX ?= g++

HW_DIR      := $(LAB4_ROOT)/src/hardware/dla/obj_dir
SRC_DIR     := $(LAB4_ROOT)/src
INCLUDE_DIR := $(LAB4_ROOT)/include
CASES_DIR   := $(LAB4_ROOT)/test/cases
BUILD_DIR   := $(TB_DIR)/build

HOST_SRC_DIR := $(LAB4_ROOT)/src/host
CPU_ORIG_DIR := $(LAB4_ROOT)/src/runtime/cpu/original

LDFLAGS := -pthread $(HW_DIR)/libVasic_wrapper.a -lm -lz

C_INCLUDES  = -I /usr/local/share/verilator/include \
              -I $(INCLUDE_DIR)/runtime \
              -I $(INCLUDE_DIR)/hal \
              -I $(HW_DIR) \
              -I $(SRC_DIR)/runtime/dla \
              -I $(HOST_SRC_DIR) \
              -I $(CPU_ORIG_DIR)

CFLAGS   = -O2 -g
C_DEFS   = -DDLA_BACKEND $(EXTRA_DEFS)

# verilated_fst_c.cpp is needed when USE_FST is enabled
VERILATOR_ROOT  := /usr/local/share/verilator
FST_EXTRA_SRCS  := $(VERILATOR_ROOT)/include/verilated_fst_c.cpp

DLA_SRCS_CPP = $(SRC_DIR)/hal/dla_hal.cpp \
               $(wildcard $(SRC_DIR)/runtime/dla/*.cpp)
DLA_SRCS_C   = $(wildcard $(SRC_DIR)/runtime/dla/*.c)

HOST_SRCS = $(HOST_SRC_DIR)/quantize.c \
            $(CPU_ORIG_DIR)/kernel_cpu.c

CASES = 0 1 2

ROOT_BUILD_DIR := $(LAB4_ROOT)/build

define BUILD_TB_CASE
$(BUILD_DIR)/tb_dla_case$(1): \
    $(TB_DIR)/tb.cpp \
    $(CASES_DIR)/case$(1)/workload.c \
    $(DLA_SRCS_CPP) $(DLA_SRCS_C) \
    $(HOST_SRCS)
	@mkdir -p $(BUILD_DIR) $(ROOT_BUILD_DIR)
	$(call msg_grey,[DLA/case$(1)] Building tb_dla_case$(1)...)
	$(CXX) $(CFLAGS) $(C_DEFS) -DCASE_NUM=$(1) \
	    '-DDLA_INFO_CSV="$(ROOT_BUILD_DIR)/dla_info_case$(1).csv"' \
	    '-DDLA_FST_DIR="$(ROOT_BUILD_DIR)/"' \
	    -I $(CASES_DIR)/case$(1) \
	    $(C_INCLUDES) \
	    $(TB_DIR)/tb.cpp \
	    $(CASES_DIR)/case$(1)/workload.c \
	    $(DLA_SRCS_CPP) $(DLA_SRCS_C) \
	    $(HOST_SRCS) \
	    $(FST_EXTRA_SRCS) \
	    $(LDFLAGS) \
	    -o $(BUILD_DIR)/tb_dla_case$(1)
	$(call msg_green,[DLA/case$(1)] Executable ready: tb_dla_case$(1))
endef

$(foreach c,$(CASES),$(eval $(call BUILD_TB_CASE,$(c))))

TB_EXES = $(foreach c,$(CASES),$(BUILD_DIR)/tb_dla_case$(c))

.PHONY: all hw_build \
        $(foreach c,$(CASES),case$(c) run$(c)) \
        clean

hw_build:
	@if [ ! -f "$(HW_DIR)/libVasic_wrapper.a" ]; then \
		$(call [DLA] Building libVasic_wrapper.a...) \
	    $(MAKE) -C $(LAB4_ROOT)/src/hardware/dla all; \
	fi

all: hw_build $(TB_EXES)
	$(call msg_green,[DLA] All testbench cases built.)

case0: hw_build $(BUILD_DIR)/tb_dla_case0
case1: hw_build $(BUILD_DIR)/tb_dla_case1
case2: hw_build $(BUILD_DIR)/tb_dla_case2

run0: case0
	$(call msg_cyan,[DLA/case0] Running...)
	$(BUILD_DIR)/tb_dla_case0

run1: case1
	$(call msg_cyan,[DLA/case1] Running...)
	$(BUILD_DIR)/tb_dla_case1

run2: case2
	$(call msg_cyan,[DLA/case2] Running...)
	$(BUILD_DIR)/tb_dla_case2

run_all: $(foreach c,$(CASES),case$(c))
	@for c in $(CASES); do \
	    printf "\n"; \
	    printf "\033[0;36m[DLA] === case$${c} ===\033[0m\n\n"; \
	    $(BUILD_DIR)/tb_dla_case$${c}; \
	done

clean:
	$(call msg_grey,[DLA] Cleaning build artifacts...)
	@rm -rf $(BUILD_DIR)

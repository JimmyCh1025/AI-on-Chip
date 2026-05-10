# src/runtime/cpu/Makefile
# Build RV64IM baremetal ELF for original and improve CPU runtimes.

LAB4_ROOT  := $(abspath ../../..)
INC_DIR    := $(LAB4_ROOT)/include/runtime
CASES_DIR  := $(LAB4_ROOT)/test/cases
THIS_DIR   := $(abspath .)

RISCV_TOOLCHAIN ?= riscv64-unknown-elf
CC = $(RISCV_TOOLCHAIN)-gcc
PICOLIBC_INC := /usr/lib/picolibc/riscv64-unknown-elf/include
PICOLIBC_LIB := /usr/lib/picolibc/riscv64-unknown-elf/lib/rv64im/lp64

# Common flags shared by all targets
COMMON_FLAGS = -march=rv64im_zicsr -mabi=lp64 -nostdlib -nostartfiles \
               -mcmodel=medany \
               -O1 -g \
               -I$(INC_DIR)

CFLAGS_ORIG = $(COMMON_FLAGS) \
              -I$(THIS_DIR)/original \
              -I$(PICOLIBC_INC)

CFLAGS_IMPR = $(COMMON_FLAGS) \
              -I$(THIS_DIR)/improve \
              -I$(PICOLIBC_INC)

LINKER  = $(THIS_DIR)/linker.ld
STARTUP = $(THIS_DIR)/startup.S

# ---------- self-test ELFs (original/improve own main.c) ----------

ORIGINAL_SRCS = $(STARTUP) \
				$(THIS_DIR)/original/kernel_cpu.c \
                $(THIS_DIR)/original/runtime_cpu.c \
                $(THIS_DIR)/original/main.c

ORIGINAL_ELF = $(THIS_DIR)/original.elf

IMPROVE_SRCS = $(STARTUP) \
			   $(THIS_DIR)/improve/kernel_cpu.c \
               $(THIS_DIR)/improve/runtime_cpu.c \
               $(THIS_DIR)/improve/main.c

IMPROVE_ELF = $(THIS_DIR)/improve.elf

# ---------- per-case ELF build helper ----------
# $(1) = case name (case0 case1 case2)
# $(2) = variant name (original / improve)
# $(3) = CFLAGS to use
# $(4) = picolibc flag (empty for original, -lc for improve)
#
# workload.c is compiled alongside main.c so that run_workload() is available.
# No -DDLA_BACKEND → workload.c uses qconv2d_relu_cpu() (CPU reference impl).
# -DCASE_ID=N lets runtime_cpu.c (improve variant) select the correct SCRATCH_SIZE.
define BUILD_CASE_ELF
$(THIS_DIR)/$(1)_$(2).elf: \
    $(STARTUP) \
	$(THIS_DIR)/$(2)/kernel_cpu.c \
    $(THIS_DIR)/$(2)/runtime_cpu.c \
    $(CASES_DIR)/$(1)/main.c \
    $(CASES_DIR)/$(1)/workload.c \
    $(LINKER)
	@printf "\033[0;37m[RUNTIME/CPU] Building $(1)_$(2).elf ...\033[0m\n"
	$(CC) $(3) \
	    -DCASE_ID=$(patsubst case%,%,$(1)) \
	    -I$(CASES_DIR)/$(1) \
	    -T$(LINKER) \
	    $(STARTUP) \
	    $(THIS_DIR)/$(2)/kernel_cpu.c \
	    $(THIS_DIR)/$(2)/runtime_cpu.c \
	    $(CASES_DIR)/$(1)/main.c \
	    $(CASES_DIR)/$(1)/workload.c \
	    $(4) \
	    -o $$@
	@printf "\033[0;37m[RUNTIME/CPU] $(1)_$(2).elf ready\033[0m\n"
endef

CASES = case0 case1 case2
CASES_FALLBACK = case_cpu_fallback_linear case_cpu_fallback_linear_relu

$(foreach c,$(CASES),$(eval $(call BUILD_CASE_ELF,$(c),original,$(CFLAGS_ORIG),)))
$(foreach c,$(CASES),$(eval $(call BUILD_CASE_ELF,$(c),improve,$(CFLAGS_IMPR),-L$(PICOLIBC_LIB) -lc)))

CASE_ELFS_ORIG = $(patsubst %,$(THIS_DIR)/%_original.elf,$(CASES))
CASE_ELFS_IMPR = $(patsubst %,$(THIS_DIR)/%_improve.elf,$(CASES))
CASE_ELFS_ALL  = $(CASE_ELFS_ORIG) $(CASE_ELFS_IMPR)

# ---------- phony targets ----------

.PHONY: all original improve cases \
        $(patsubst %,%_original,$(CASES)) \
        $(patsubst %,%_improve,$(CASES)) \
        clean

all: original improve cases

original: $(ORIGINAL_ELF)

$(ORIGINAL_ELF): $(ORIGINAL_SRCS) $(LINKER)
	@echo -e "\033[0;37m[RUNTIME/CPU] Building original.elf ...\033[0m"
	$(CC) $(CFLAGS_ORIG) -T$(LINKER) $(ORIGINAL_SRCS) -o $@
	@echo -e "\033[0;37m[RUNTIME/CPU] original.elf ready\033[0m"

improve: $(IMPROVE_ELF)

$(IMPROVE_ELF): $(IMPROVE_SRCS) $(LINKER)
	@echo -e "\033[0;37m[RUNTIME/CPU] Building improve.elf ...\033[0m"
	$(CC) $(CFLAGS_IMPR) -T$(LINKER) $(IMPROVE_SRCS) \
	    -L$(PICOLIBC_LIB) -lc \
	    -o $@
	@echo -e "\033[0;37m[RUNTIME/CPU] improve.elf ready\033[0m"

cases: $(CASE_ELFS_ALL)
	@echo -e "\033[0;37m[RUNTIME/CPU] All case ELFs ready.\033[0m"

$(patsubst %,%_original,$(CASES)): %_original : $(THIS_DIR)/%_original.elf
$(patsubst %,%_improve, $(CASES)): %_improve  : $(THIS_DIR)/%_improve.elf

define BUILD_FALLBACK_ELF
$(THIS_DIR)/case_cpu_fallback_$(1)_$(2).elf: \
    $(STARTUP) \
    $(THIS_DIR)/$(2)/kernel_cpu.c \
    $(THIS_DIR)/$(2)/runtime_cpu.c \
    $(CASES_DIR)/case_cpu_fallback/main.c \
    $(CASES_DIR)/case_cpu_fallback/$(6) \
    $(LINKER)
	@printf "\033[0;37m[RUNTIME/CPU] Building case_cpu_fallback_$(1)_$(2).elf ...\033[0m\n"
	$(CC) $(3) $(4) \
	    -I$(CASES_DIR)/case_cpu_fallback \
	    -T$(LINKER) \
	    $(STARTUP) \
	    $(THIS_DIR)/$(2)/kernel_cpu.c \
	    $(THIS_DIR)/$(2)/runtime_cpu.c \
	    $(CASES_DIR)/case_cpu_fallback/main.c \
	    $(CASES_DIR)/case_cpu_fallback/$(6) \
	    $(5) \
	    -o $$@
	@printf "\033[0;37m[RUNTIME/CPU] case_cpu_fallback_$(1)_$(2).elf ready\033[0m\n"
endef

$(foreach v,original improve,$(eval $(call BUILD_FALLBACK_ELF,linear,$(v),$(if $(filter improve,$(v)),$(CFLAGS_IMPR),$(CFLAGS_ORIG)),-DLINEAR -DCASE_ID=3,$(if $(filter improve,$(v)),-L$(PICOLIBC_LIB) -lc,),workload_linear.c)))
$(foreach v,original improve,$(eval $(call BUILD_FALLBACK_ELF,linear_relu,$(v),$(if $(filter improve,$(v)),$(CFLAGS_IMPR),$(CFLAGS_ORIG)),-DLINEAR_RELU -DCASE_ID=4,$(if $(filter improve,$(v)),-L$(PICOLIBC_LIB) -lc,),workload_linear_relu.c)))

CASE_FALLBACK_ELFS_ORIG = $(patsubst %,$(THIS_DIR)/%_original.elf,$(CASES_FALLBACK))
CASE_FALLBACK_ELFS_IMPR = $(patsubst %,$(THIS_DIR)/%_improve.elf,$(CASES_FALLBACK))
CASE_FALLBACK_ELFS_ALL  = $(CASE_FALLBACK_ELFS_ORIG) $(CASE_FALLBACK_ELFS_IMPR)

.PHONY: $(patsubst %,%_original,$(CASES_FALLBACK)) \
        $(patsubst %,%_improve, $(CASES_FALLBACK)) \
        fallback_cases

fallback_cases: $(CASE_FALLBACK_ELFS_ALL)
	@echo -e "\033[0;37m[RUNTIME/CPU] All case_cpu_fallback ELFs ready.\033[0m"

$(patsubst %,%_original,$(CASES_FALLBACK)): %_original : $(THIS_DIR)/%_original.elf
$(patsubst %,%_improve, $(CASES_FALLBACK)): %_improve  : $(THIS_DIR)/%_improve.elf

clean:
	@echo -e "\033[0;37m[RUNTIME/CPU] Cleaning ...\033[0m"
	@rm -f $(ORIGINAL_ELF) $(IMPROVE_ELF) $(CASE_ELFS_ALL) $(CASE_FALLBACK_ELFS_ALL)
	@rm -f $(THIS_DIR)/case_cpu_fallback_*.elf

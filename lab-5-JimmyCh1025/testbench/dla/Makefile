# DLA Verilator testbench

APP     := main
ROOT    := $(abspath ../..)
OBJ_DIR := obj
LOG_DIR := log

OUTPUT_DIR := $(ROOT)/output
HW_DIR     := $(ROOT)/simulation/hardware
RUNTIME_DIR := $(ROOT)/simulation/software
HAL_SRC_DIR := $(RUNTIME_DIR)/src/hal
DLA_SRC_DIR := $(RUNTIME_DIR)/src/eyeriss/dla
CPU_SRC_DIR := $(RUNTIME_DIR)/src/eyeriss/cpu
CSOURCE_DIR := $(ROOT)/Csource
HW_LIB     := $(HW_DIR)/obj_dir/libVasic_wrapper.a

CC  ?= gcc
CXX ?= g++
QUIET ?= @
ifeq ($(QUIET),0)
  QUIET :=
endif
export PATH := /opt/verilator-5.034/bin:$(PATH)

CLASS ?= 4
INDEX ?= 9
RUN_ARGS := -i ../../output/bin/input.bin -w dummy

DEFS :=
ifeq ($(DLA_INFO),1)
  DEFS += -DDLA_INFO
endif
ifeq ($(DEBUG),1)
  DEFS += -DDEBUG
endif
ifeq ($(USE_VCD),1)
  DEFS += -DUSE_VCD
endif

CPPFLAGS := $(DEFS) \
            -I/usr/local/share/verilator/include \
            -I$(RUNTIME_DIR)/include/eyeriss \
            -I$(RUNTIME_DIR)/include/hal \
            -I$(CPU_SRC_DIR) \
            -I$(HW_DIR) \
            -I$(HW_DIR)/obj_dir \
            -I$(OUTPUT_DIR)/lib_objects \
            -I$(CSOURCE_DIR) \
            -I.
CFLAGS   ?=
CXXFLAGS ?=
LDLIBS   := $(OUTPUT_DIR)/lib_dla.so \
            -Wl,-rpath,$(OUTPUT_DIR) \
            -pthread -rdynamic -L$(HW_DIR)/obj_dir -lVasic_wrapper -lz

SRC_C := $(wildcard $(CSOURCE_DIR)/*.c) \
         $(wildcard $(CPU_SRC_DIR)/*.c) \
         $(wildcard $(HAL_SRC_DIR)/*.c) \
         $(wildcard $(DLA_SRC_DIR)/*.c)
SRC_CPP := main.cpp \
           $(wildcard $(CSOURCE_DIR)/*.cpp) \
           $(wildcard $(HAL_SRC_DIR)/*.cpp) \
           $(wildcard $(DLA_SRC_DIR)/*.cpp)
OBJ := $(addprefix $(OBJ_DIR)/,$(notdir $(SRC_C:.c=.o) $(SRC_CPP:.cpp=.o)))

vpath %.c   $(sort $(dir $(SRC_C)))
vpath %.cpp $(sort $(dir $(SRC_CPP)))

.PHONY: all clean help build_hw test nWave

all: $(APP)

help:
	@echo "Usage: make <target> [DLA_INFO=1] [DEBUG=1] [USE_VCD=1] [CLASS=<num> INDEX=<num>]"
	@echo ""
	@echo "Targets:"
	@echo "  all       Build Verilated DLA and host testbench"
	@echo "  test      Run one DLA inference, default CLASS=4 INDEX=9"
	@echo "  build_hw  Build the Verilated DLA library"
	@echo "  nWave     Open waveform viewer workspace"
	@echo "  clean     Remove generated files"

build_hw: $(HW_LIB)

$(HW_LIB):
	$(QUIET)$(MAKE) -C $(HW_DIR) all

$(APP): $(OBJ) $(HW_LIB) | $(OBJ_DIR)
	$(QUIET)echo "LD $@"
	$(QUIET)cd $(OBJ_DIR) && $(CXX) $(notdir $(OBJ)) $(LDLIBS) -o ../$(APP)

$(OBJ): | $(HW_LIB)

$(OBJ_DIR)/%.o: %.c | $(OBJ_DIR)
	$(QUIET)echo "CC $(notdir $@)"
	$(QUIET)$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(QUIET)echo "CXX $(notdir $@)"
	$(QUIET)$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(OBJ_DIR) $(LOG_DIR):
	$(QUIET)mkdir -p $@

test: $(APP) | $(LOG_DIR)
	$(QUIET)echo "Run DLA test"
	$(QUIET)bash -c "./$(APP) $(RUN_ARGS) -c $(CLASS) -n $(INDEX) \
	  1> >(tee $(LOG_DIR)/out.log) 2> >(tee $(LOG_DIR)/err.log >&2)"

nWave:
	$(QUIET)mkdir -p nWave_log
	$(QUIET)cd nWave_log && nWave &

clean:
	$(QUIET)rm -rf $(OBJ_DIR) $(LOG_DIR) nWave_log
	$(QUIET)rm -f $(APP) *.o *.vcd *.fsdb *.log *.rc *.csv
	$(QUIET)$(MAKE) -C $(HW_DIR) clean

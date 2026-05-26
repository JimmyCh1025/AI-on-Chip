# CPU-only host reference testbench

APP      := main
ROOT     := $(abspath ../..)
OBJ_DIR  := obj
LOG_DIR  := log
MASSIF_DIR := massif_out

OUTPUT_DIR := $(ROOT)/output
RUNTIME_DIR := $(ROOT)/simulation/software
CPU_SRC_DIR := $(RUNTIME_DIR)/src/eyeriss/cpu
CSOURCE_DIR := $(ROOT)/Csource

CC       ?= gcc
CXX      ?= g++
VALGRIND ?= valgrind
QUIET    ?= @
ifeq ($(QUIET),0)
  QUIET :=
endif

CLASS ?= 4
INDEX ?= 9
RUN_ARGS := -i ../../output/bin/input.bin -w dummy

CPPFLAGS := -DCPU_ONLY \
            -I$(RUNTIME_DIR)/include/eyeriss \
            -I$(CPU_SRC_DIR) \
            -I$(OUTPUT_DIR)/lib_objects \
            -I$(CSOURCE_DIR) \
            -I.
CFLAGS   ?=
CXXFLAGS ?=
LDLIBS   := $(OUTPUT_DIR)/lib_cpu.so -Wl,-rpath,$(OUTPUT_DIR) -pthread -lm

SRC_C   := $(wildcard $(CSOURCE_DIR)/*.c) $(wildcard $(CPU_SRC_DIR)/*.c)
SRC_CPP := main.cpp
OBJ     := $(addprefix $(OBJ_DIR)/,$(notdir $(SRC_C:.c=.o) $(SRC_CPP:.cpp=.o)))

vpath %.c   $(sort $(dir $(SRC_C)))
vpath %.cpp $(sort $(dir $(SRC_CPP)))

.PHONY: all clean help test test_full valgrind valgrind_full

all: $(APP)

help:
	@echo "Usage: make <target> [CLASS=<num> INDEX=<num>]"
	@echo ""
	@echo "Targets:"
	@echo "  all            Build CPU-only host executable"
	@echo "  test           Run one inference, default CLASS=4 INDEX=9"
	@echo "  test_full      Run all 100 test images"
	@echo "  valgrind       Run one inference under Valgrind Massif"
	@echo "  valgrind_full  Run all 100 test images under Valgrind Massif"
	@echo "  clean          Remove generated files"

$(APP): $(OBJ) | $(OBJ_DIR)
	$(QUIET)echo "LD $@"
	$(QUIET)cd $(OBJ_DIR) && $(CXX) $(notdir $(OBJ)) $(LDLIBS) -o ../$(APP)

$(OBJ_DIR)/%.o: %.c | $(OBJ_DIR)
	$(QUIET)echo "CC $(notdir $@)"
	$(QUIET)$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(QUIET)echo "CXX $(notdir $@)"
	$(QUIET)$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(OBJ_DIR) $(LOG_DIR) $(MASSIF_DIR):
	$(QUIET)mkdir -p $@

test: $(APP) | $(LOG_DIR)
	$(QUIET)echo "Run CPU-only test"
	$(QUIET)bash -c "./$(APP) $(RUN_ARGS) -c $(CLASS) -n $(INDEX) \
	  1> >(tee $(LOG_DIR)/out.log) 2> >(tee $(LOG_DIR)/err.log >&2)"

test_full:
	$(QUIET)$(MAKE) test CLASS=-1 INDEX=-1

valgrind: $(APP) | $(LOG_DIR) $(MASSIF_DIR)
	$(QUIET)echo "Run CPU-only test under Valgrind Massif"
	$(QUIET)$(VALGRIND) --tool=massif \
	  --log-file=$(LOG_DIR)/$(APP)_massif.log \
	  --heap=yes \
	  --stacks=yes \
	  --time-unit=i \
	  --detailed-freq=1 \
	  --max-snapshots=1000 \
	  --ignore-fn=fopen --ignore-fn=fread --ignore-fn=fwrite --ignore-fn=load_bin_data \
	  --massif-out-file=$(MASSIF_DIR)/massif.out.%p_$(APP) \
	  ./$(APP) $(RUN_ARGS) -c $(CLASS) -n $(INDEX)

valgrind_full:
	$(QUIET)$(MAKE) valgrind CLASS=-1 INDEX=-1

clean:
	$(QUIET)rm -rf $(OBJ_DIR) $(LOG_DIR) $(MASSIF_DIR)
	$(QUIET)rm -f $(APP) *.o *.vcd *.fsdb *.log *.rc *.csv

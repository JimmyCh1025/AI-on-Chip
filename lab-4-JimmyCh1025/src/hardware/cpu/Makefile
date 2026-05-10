LAB4_ROOT := $(abspath ../..)
RTL_DIR   := $(abspath rtl)
OBJ_DIR   := $(abspath obj_dir)

VERILATOR  = verilator
VL_FLAGS   = --cc --trace-fst --build -j 0
VL_FLAGS  += -I$(RTL_DIR)
VL_FLAGS  += --Mdir $(OBJ_DIR)
VL_FLAGS  += --timescale 1ns/1ps
VL_FLAGS  += --relative-includes
VL_FLAGS  += -Wno-SYMRSVDWORD
VL_FLAGS  += -Wno-DECLFILENAME
VL_FLAGS  += -Wno-WIDTHTRUNC
VL_FLAGS  += -Wno-UNOPTFLAT
VL_FLAGS  += -Wno-CASEINCOMPLETE

# Top module
VL_TOP = NutShell

RTL_SV_TOP  = $(RTL_DIR)/NutShell.sv
RTL_SV_REST = $(shell find $(RTL_DIR) -name "*.sv" \
                ! -name "NutShell.sv"    \
                ! -name "NutShellSim.sv" \
                ! -name "SimTop.sv"      \
                ! -name "SimMMIO.sv" 2>/dev/null)
RTL_V       = $(shell find $(RTL_DIR) -name "*.v" 2>/dev/null)
ASSERT_V    = $(LAB4_ROOT)/../test/testbench/cpu/assert.v

LIB = $(OBJ_DIR)/libVNutShell.a

.PHONY: all clean

all: $(LIB)

$(LIB): $(RTL_SV_TOP) $(RTL_SV_REST) $(RTL_V) $(ASSERT_V)
	@mkdir -p $(OBJ_DIR)
	@echo "[HW/CPU] Verilating NutShell..."
	$(VERILATOR) $(VL_FLAGS) \
	    --top-module $(VL_TOP) \
	    $(RTL_SV_TOP) $(RTL_SV_REST) $(RTL_V) $(ASSERT_V)
	@echo "[HW/CPU] libVNutShell.a ready at $(OBJ_DIR)"

clean:
	@echo "[HW/CPU] Cleaning obj_dir..."
	@rm -rf $(OBJ_DIR)

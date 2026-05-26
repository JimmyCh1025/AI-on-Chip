CC = gcc
CFLAGS = -g -Wall
TARGET = massif_test
SRC = massif_test.c
OUTPUT_DIR = massif_out
OUTPUT = $(OUTPUT_DIR)/massif.out.$(TARGET)

.PHONY: all clean massif ms_print

all: $(OUTPUT_DIR)/$(TARGET)

$(OUTPUT_DIR)/$(TARGET): $(SRC)
	mkdir -p $(OUTPUT_DIR)
	$(CC) $(CFLAGS) -o $@ $(SRC)

clean:
	rm -rf $(OUTPUT_DIR)

$(OUTPUT): massif
	@echo "Massif output file: $(OUTPUT)"

massif: $(OUTPUT_DIR)/$(TARGET)
	mkdir -p $(OUTPUT_DIR)
	valgrind --tool=massif \
	--heap=yes \
	--stacks=yes \
	--time-unit=i \
	--detailed-freq=1 \
	--max-snapshots=1000 \
	--massif-out-file=$(OUTPUT) ./$(OUTPUT_DIR)/$(TARGET)

ms_print: $(OUTPUT)
	ms_print $(OUTPUT) > $(OUTPUT_DIR)/massif_output.txt

`include "AXI_define.svh"
`include "ASIC.svh"
`define FIFO_WORDS_SIZE 64
`define FIFO_ADDR_WIDTH 6
`define FIFO_COUNT_WIDTH 7

module DMA_FIFO (
    input clk, // System clock
    input rst, // System reset (active high)
    input push_i,
    input pop_i,
    input [`DATA_BITS-1:0] data_i,
    output logic [`DATA_BITS-1:0] data_o,
    output logic full,
    output logic empty
);

    // FIFO memory
    logic [`DATA_BITS-1:0] fifo_mem[`FIFO_WORDS_SIZE-1:0];

    // Read and write pointers
    logic [`FIFO_ADDR_WIDTH-1:0] write_ptr;
    logic [`FIFO_ADDR_WIDTH-1:0] read_ptr;

    // FIFO occupancy counter
    logic [`FIFO_COUNT_WIDTH-1:0] fifo_count;

    // Write logic
    always_ff @(posedge clk) begin
        if (rst) begin
            write_ptr <= `FIFO_ADDR_WIDTH'd0;
        end else if (push_i && !full) begin
            fifo_mem[write_ptr] <= data_i;
            write_ptr <= write_ptr + `FIFO_ADDR_WIDTH'd1; // back to 0 when overflow
        end else begin
            write_ptr <= write_ptr;
        end
    end

    // Read logic
    assign data_o = fifo_mem[read_ptr];

    always_ff @(posedge clk) begin
        if (rst) begin
            read_ptr <= `FIFO_ADDR_WIDTH'd0;
        end else if (pop_i && !empty) begin
            read_ptr <= read_ptr + `FIFO_ADDR_WIDTH'd1; // back to 0 when overflow
        end else begin
            read_ptr <= read_ptr;
        end
    end

    // FIFO count logic
    always_ff @(posedge clk) begin
        if (rst) begin
            fifo_count <= `FIFO_COUNT_WIDTH'd0;
        end else begin
            case ({push_i && !full, pop_i && !empty})
                2'b10: fifo_count <= fifo_count + `FIFO_COUNT_WIDTH'd1; // Write only
                2'b01: fifo_count <= fifo_count - `FIFO_COUNT_WIDTH'd1; // Read only
                default: fifo_count <= fifo_count;   // No operation or simultaneous read/write
            endcase
        end
    end

    // Full and empty flags
    assign full = (fifo_count == `FIFO_COUNT_WIDTH'd`FIFO_WORDS_SIZE);
    assign empty = (fifo_count == `FIFO_COUNT_WIDTH'd0);

endmodule

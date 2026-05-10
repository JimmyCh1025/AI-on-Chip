/*
    DMA controller
    (without address alignment, ignore addr[1:0])
*/
`include "AXI_define.svh"
`include "ASIC.svh"
`define MAX_BURST_LEN 64
`define MAX_BURST_LEN_ 63

module DMA_controller (
    input clk, // System clock
    input rst, // System reset (active high)
    input DMAEN, // Enable the DMA
    input [`GLB_ADDR_BITS-1:0] DMAGLB_ADDR, // GLB address of DMA
    input [`AXI_ADDR_BITS-1:0] DMADRAM_ADDR, // DRAM address of DMA
    input [`AXI_ADDR_BITS-1:0] DMALEN, // Total length of the data
    output logic DMA_done,  // DMA done

    /* AXI control */
    output logic [`AXI_ADDR_BITS-1:0] dram_addr,
    output logic [`AXI_LEN_BITS-1:0] burst_len,
    output logic [`AXI_SIZE_BITS-1:0] burst_size,
    output logic AXI_enable,
    input AXI_done,

    /* GLB control */
    output logic [`GLB_ADDR_BITS-1:0] glb_addr,
    output logic GLB_enable,
    input GLB_done
);

enum logic [1:0] {IDLE, BUSY, DONE_AXI,DONE_S} DMA_state, DMA_state_next;

logic [`AXI_ADDR_BITS-1:0] word_len_mask;
logic [`AXI_ADDR_BITS-1:0] word_len_reg;
logic [`AXI_ADDR_BITS-1:0] dram_addr_next, word_len_reg_next;
logic [`GLB_ADDR_BITS-1:0] glb_addr_next;
logic [`AXI_LEN_BITS-1:0] burst_len_next;


assign DMA_done = (DMA_state == DONE_S)?1'b1:1'b0;
// assign word_len_mask = (DMALEN[1:0] == 2'd0)? {2'd0,DMALEN[`AXI_ADDR_BITS-1:2]}:{2'd0,DMALEN[`AXI_ADDR_BITS-1:2]} + `AXI_ADDR_BITS'd1; // word alignment (32 bits)
assign word_len_mask = DMALEN;

/* DMA controller FSM */
always_ff @(posedge clk) begin
    if(rst) begin
        dram_addr <= `AXI_ADDR_BITS'd0;
        glb_addr <= `GLB_ADDR_BITS'd0;
        word_len_reg <= `AXI_ADDR_BITS'd0;
        DMA_state <= IDLE;
        burst_len <= `AXI_LEN_BITS'd0;
    end else begin
        dram_addr <= dram_addr_next;
        glb_addr <= glb_addr_next;
        word_len_reg <= word_len_reg_next;
        DMA_state <= DMA_state_next;
        burst_len <= burst_len_next;
    end
end

/* DMA controller Comb. */
always_comb begin
    dram_addr_next = dram_addr;
    glb_addr_next = glb_addr;
    word_len_reg_next = word_len_reg;
    DMA_state_next = DMA_state;
    burst_len_next = burst_len;

    burst_size = `AXI_SIZE_WORD;
    AXI_enable = 1'b0;
    GLB_enable = 1'b0;

    case (DMA_state)
        IDLE: begin
            if(DMAEN) begin // got task information
                dram_addr_next = DMADRAM_ADDR;
                glb_addr_next = DMAGLB_ADDR;
                DMA_state_next = BUSY;
                if(word_len_mask < `AXI_ADDR_BITS'd`MAX_BURST_LEN) begin
                    burst_len_next = word_len_mask[`AXI_LEN_BITS-1:0] - `AXI_LEN_BITS'd1;
                    word_len_reg_next = `AXI_ADDR_BITS'd0; // remained
                end else begin
                    burst_len_next = `AXI_LEN_BITS'd`MAX_BURST_LEN_;
                    word_len_reg_next = word_len_mask - `AXI_ADDR_BITS'd`MAX_BURST_LEN; // remained
                end
            end
        end
        BUSY: begin
            AXI_enable = 1'b1;
            GLB_enable = 1'b1;
            if(AXI_done) begin
                dram_addr_next = dram_addr_next + {30'd`MAX_BURST_LEN,2'b00};
                if(word_len_reg == `AXI_ADDR_BITS'd0) begin // all done
                    DMA_state_next = DONE_AXI;
                end else if(word_len_reg < `AXI_ADDR_BITS'd`MAX_BURST_LEN) begin
                    burst_len_next = word_len_reg[`AXI_LEN_BITS-1:0] - `AXI_LEN_BITS'd1;
                    word_len_reg_next = `AXI_ADDR_BITS'd0; // remained
                end else begin
                    burst_len_next = `AXI_LEN_BITS'd`MAX_BURST_LEN_;
                    word_len_reg_next = word_len_reg - `AXI_ADDR_BITS'd`MAX_BURST_LEN; // remained
                end
            end
        end
        DONE_AXI: begin
            GLB_enable = 1'b1;
            if(GLB_done)begin
                DMA_state_next = DONE_S;
            end
        end
        DONE_S: begin
            // if(!DMAEN)begin
                DMA_state_next = IDLE;
            // end
        end
        default: DMA_state_next = IDLE;
    endcase
end

endmodule

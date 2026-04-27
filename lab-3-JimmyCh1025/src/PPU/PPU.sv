`include "src/PPU/PostQuant.sv"
`include "src/PPU/Maxpool_Qint8.sv"
`include "src/PPU/ReLU_Qint8.sv"
`include "define.svh"

module PPU (
    input clk,
    input rst,
    input [`DATA_BITS-1:0] data_in,
    input [5:0] scaling_factor,
    input maxpool_en,
    input maxpool_init,
    input relu_sel,
    input relu_en,
    output logic[7:0] data_out
);

    logic [7:0] data_post_quant, data_post_quant_delayed;
    logic [7:0] data_maxpool;
    logic [7:0] data_sel;

    PostQuant postquant (
        .data_in       (data_in),
        .scaling_factor(scaling_factor),
        .data_out      (data_post_quant)
    );

    Maxpool_Qint8 maxpool (
        .clk     (clk),
        .rst     (rst),
        .en      (maxpool_en),
        .init    (maxpool_init),
        .data_in (data_post_quant),
        .data_out(data_maxpool)
    );

    ReLU_Qint8 relu (
        .en      (relu_en),
        .data_in (data_sel),
        .data_out(data_out)
    );

    
    
    always_ff @(posedge clk or posedge rst) begin
        if (rst)
            data_post_quant_delayed <= 8'd0;
        else     
            data_post_quant_delayed <= data_post_quant;
    end

    assign data_sel = relu_sel? data_maxpool : data_post_quant_delayed;

endmodule
`include "ASIC.svh"

module PPU (
    input clk,
    input rst,
    input [`DATA_BITS-1:0] data_in,
    input [5:0] scaling_factor,
    input maxpool_en,
    input maxpool_init,
    input relu_en,
    input relu_sel,
    output logic[7:0] data_out
);

/* PostQuant -> MaxPool -> ReLU */

logic [7:0] pq_data_out;
logic [7:0] pq_data_out_reg;
logic [7:0] maxpool_data_out;
logic [7:0] relu_data_in;

post_quant post_quant_0(
    .data_in(data_in),
    .scaling_factor(scaling_factor),
    .data_out(pq_data_out)
);

always_ff @( posedge clk ) begin
    if(rst) begin
        pq_data_out_reg <= 8'd0;
    end else begin
        pq_data_out_reg <= pq_data_out;
    end
end

Comparator_Qint8 Comparator_Qint8_0(
    .clk(clk),
    .rst(rst),
    .en(maxpool_en),
    .init(maxpool_init),
    .data_in(pq_data_out),
    .data_out(maxpool_data_out)
);

assign relu_data_in = (relu_sel)? maxpool_data_out: pq_data_out_reg;

ReLU_Qint8 ReLU_Qint8_0(
    .en(relu_en),
    .data_in(relu_data_in),
    .data_out(data_out)
);

endmodule

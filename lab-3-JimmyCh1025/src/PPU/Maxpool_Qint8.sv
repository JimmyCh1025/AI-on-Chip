module Maxpool_Qint8 (
    input clk,
    input rst,
    input en,
    input init,
    input logic [7:0] data_in,
    output logic [7:0] data_out
);

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            data_out <= 8'd0;
        end else if (init) begin
            data_out <= data_in;
        end else if (en) begin
            data_out <= (data_in > data_out)? data_in : data_out;
        end
    end

endmodule

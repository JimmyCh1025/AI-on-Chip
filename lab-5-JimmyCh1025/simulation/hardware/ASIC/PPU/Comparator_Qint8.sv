module Comparator_Qint8 (
    input clk,
    input rst,
    input en,
    input init,
    input logic [7:0] data_in,
    output logic [7:0] data_out
);
    logic [7:0] data_max, data_max_next;
    logic bigger;

    always_ff @( posedge clk ) begin
        if(rst)begin
            data_max <= 8'd0;
        end else begin
            data_max <= data_max_next;
        end
    end

    always_comb begin
        bigger = (data_max < data_in)?1'b1:1'b0;
        if(en)begin
            if(init) begin
                data_max_next = data_in;
            end else begin
                if(bigger) begin
                    data_max_next = data_in;
                end else begin
                    data_max_next = data_max;
                end
            end
        end else begin
            data_max_next = data_max;
        end
        data_out = data_max;
    end

endmodule

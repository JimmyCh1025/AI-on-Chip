module ReLU_Qint8 (
    input en,
    input [7:0] data_in,
    output logic [7:0] data_out
);
    localparam ZERO_POINT = 8'd128;

    always_comb begin
        if (en) begin
            case (data_in[7])
                1'd0:
                    data_out = ZERO_POINT;
                1'd1:
                    data_out = data_in;
            endcase
        end else begin
            data_out = data_in;
        end
    end

endmodule
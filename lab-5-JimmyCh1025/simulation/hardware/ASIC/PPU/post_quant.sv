`include "ASIC.svh"
module post_quant (
    input [`DATA_BITS-1:0] data_in,
    input [5:0] scaling_factor,
    output logic [7:0] data_out
);

logic [`DATA_BITS-1:0]  data_shifted;
logic overflow_pos;
logic overflow_neg;


always_comb begin
    data_shifted = $signed(data_in) >>> scaling_factor; // scale
    overflow_pos = (~data_shifted[`DATA_BITS-1]) && (|(data_shifted[`DATA_BITS-2:7]));
    overflow_neg =  (data_shifted[`DATA_BITS-1]) && ~(&(data_shifted[`DATA_BITS-2:7]));
    data_out = (overflow_pos)? (8'hff): ((overflow_neg)? 8'd0: {~data_shifted[7],data_shifted[6:0]});
end

endmodule

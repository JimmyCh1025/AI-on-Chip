`include "define.svh"
module PostQuant (
    input [`DATA_BITS-1:0] data_in,
    input [5:0] scaling_factor,
    output logic [7:0] data_out
);

    logic [7:0] cmp_data_tmp;
    logic signed [`DATA_BITS-1:0] tmp;
    logic sign_bit = data_in[`DATA_BITS-1];

    // comparator
    always_comb begin
        // if data is neg, then clamp to 0
        cmp_data_tmp = tmp[7:0] & ~{8{sign_bit}}; 
        
        // if data is pos && data > 256, then 256 
        data_out = cmp_data_tmp | {8{|tmp[`DATA_BITS-2:8] & ~sign_bit}}; 
    end

    // +128
    always_comb begin
        tmp = (data_in >>> scaling_factor);
        // if value [128~255], then carry one bit
        if (!sign_bit && tmp[7]) begin
            tmp[8:7] = 2'b10;
        end else begin
            tmp[7] = ~tmp[7];
        end
        
    end

endmodule



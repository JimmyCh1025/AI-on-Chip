module GON_MulticastController #(
    parameter ID_SIZE = 6,
    parameter DATA_SIZE = 32
)(
    input clk,
    input rst,

    // config id
    input set_id,
    input [ID_SIZE - 1:0] id_in,
    output logic [ID_SIZE - 1:0] id,

    // tag
    input [ID_SIZE - 1:0] tag,

    input valid_in,
    output logic valid_out,
    input ready_in,
    output logic ready_out
);
always_ff @( posedge clk ) begin
    if (rst) id <= 'd0;
    else begin
        if (set_id) id <= id_in;
    end
end

assign valid_out = (tag == id)? valid_in: 1'b0;
assign ready_out = (tag == id)? ready_in: 1'b0;

endmodule

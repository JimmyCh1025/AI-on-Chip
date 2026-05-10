//  `include "src/PE_array/GON/GON_MulticastController.sv"
 module GON_Bus #(
    parameter NUMS_MASTER = 8,
    parameter TAG_SIZE = 6,
    parameter DATA_SIZE = 32
) (
    input clk,
    input rst,
    input [TAG_SIZE - 1:0] tag,

    input [NUMS_MASTER - 1:0] master_valid,
    input [NUMS_MASTER * DATA_SIZE - 1:0] master_data,
    output logic [NUMS_MASTER - 1:0] master_ready,

    output logic slave_valid,
    input slave_ready,
    output logic [DATA_SIZE - 1:0] slave_data,

    // Config
    input set_id,
    input [TAG_SIZE - 1:0] ID_scan_in,
    output logic [TAG_SIZE - 1 :0] ID_scan_out
 );

// tag -> The tag represents the PE ID to which the input data should be sent.
// ID ->　The ID represents the identifier number of the PE

logic [NUMS_MASTER - 1:0] mc_valid;
logic [NUMS_MASTER:0] select_valid;
/* verilator lint_off UNOPTFLAT */
logic [DATA_SIZE - 1:0] select_data [0:NUMS_MASTER];
logic [TAG_SIZE - 1:0] ID_scan_chain [0:NUMS_MASTER];
genvar i;

assign select_data[0] = 'd0;
assign ID_scan_chain[NUMS_MASTER] = ID_scan_in;

generate
for (i = 0; i < NUMS_MASTER; i++) begin : GON_MC
   GON_MulticastController #(
      .ID_SIZE(TAG_SIZE),
      .DATA_SIZE(DATA_SIZE)
   ) MC_i (
      .clk(clk),
      .rst(rst),
      .set_id(set_id),
      .id_in(ID_scan_chain[i+1]),
      .id(ID_scan_chain[i]),
      .tag(tag),
      .valid_in(master_valid[i]),
      .valid_out(mc_valid[i]),
      .ready_in(slave_ready),
      .ready_out(master_ready[i])
   );
   assign select_data[i+1] = (mc_valid[i])? select_data[i] | master_data[(i+1)*DATA_SIZE-1:i*DATA_SIZE]: select_data[i];
end
endgenerate
/* verilator lint_on UNOPTFLAT */
// Master I/O: Bus <-> PE
assign slave_valid = |mc_valid;
assign slave_data = (slave_valid)? select_data[NUMS_MASTER]: 'd0;
// Config
assign ID_scan_out = ID_scan_chain[0];

endmodule

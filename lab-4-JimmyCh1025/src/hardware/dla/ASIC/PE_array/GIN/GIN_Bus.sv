 module GIN_Bus #(
    parameter NUMS_SLAVE = 8,
    parameter TAG_SIZE = 6,
    parameter DATA_SIZE = 32
) (
    input clk,
    input rst,
    input [TAG_SIZE - 1:0] tag,

    input master_valid,
    input [DATA_SIZE - 1:0] master_data,
    output logic master_ready,

    input [NUMS_SLAVE - 1:0] slave_ready,
    output logic [NUMS_SLAVE - 1:0] slave_valid,
    output logic [NUMS_SLAVE*DATA_SIZE - 1:0] slave_data,
    // Config
    input set_id,
    input [TAG_SIZE - 1:0] ID_scan_in,
    output logic [TAG_SIZE - 1 :0] ID_scan_out
 );

// tag -> The tag represents the PE ID to which the input data should be sent.
// ID ->　The ID represents the identifier number of the PE

// 1 indicates that the data for this round has already been received
// 0 indicates it has not yet been received.
logic [NUMS_SLAVE - 1:0] valid_mask;
logic [NUMS_SLAVE - 1:0] mc_valid;
logic [NUMS_SLAVE - 1:0] mc_ready;
logic [TAG_SIZE - 1:0] ID_scan_chain [0:NUMS_SLAVE];

integer j;

always_ff @( posedge clk ) begin
   if (rst) valid_mask <= 'd0;
   else if (master_valid && master_ready) valid_mask <= 'd0;
   else begin
      for (j = 0; j < NUMS_SLAVE; j = j + 1) begin
         if (mc_valid[j] && mc_ready[j]) valid_mask[j] <= 1'd1;
      end
   end
end

generate
genvar i;
for (i = 0; i < NUMS_SLAVE; i++) begin : GIN_MC
   GIN_MulticastController #(
      .TAG_SIZE(TAG_SIZE),
      .DATA_SIZE(DATA_SIZE)
   ) MC (
      .clk(clk),
      .rst(rst),
      .set_id(set_id),
      .id_in(ID_scan_chain[i+1]),
      .id(ID_scan_chain[i]),
      .tag(tag),
      .valid_in(master_valid),
      .valid_out(mc_valid[i]),
      .ready_in(slave_ready[i]),
      .ready_out(mc_ready[i])
   );
end
endgenerate

assign ID_scan_chain[NUMS_SLAVE] = ID_scan_in;

// Slave I/O: SRAM <-> Bus
// All master are ready -> ready = 1
assign master_ready = &(mc_ready | valid_mask);

// Master I/O: Bus <-> PE
assign slave_valid = mc_valid & ~valid_mask;
assign slave_data = {NUMS_SLAVE{master_data}};

// Config
assign ID_scan_out = ID_scan_chain[0];

endmodule

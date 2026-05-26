// `include "src/PE_array/GIN/GIN_Bus.sv"
// `include "src/PE_array/GIN/GIN_MulticastController.sv"

module GIN #(
    parameter   NUMS_XBUS    = 6,
                NUMS_XPE     = 8,
                XID_SIZE    = 6,
                YID_SIZE    = 3,
                DATA_SIZE    = 32
) (
    input clk,
    input rst,

    // Slave SRAM <-> GIN
    input GIN_valid,
    output logic GIN_ready,
    input [DATA_SIZE - 1:0] GIN_data,

    /* Controller <-> GIN */
    input [XID_SIZE - 1:0] tag_X,
    input [YID_SIZE - 1:0] tag_Y,

    /* config */
    input set_XID,
    input [XID_SIZE - 1:0] XID_scan_in,
    input set_YID,
    input [YID_SIZE - 1:0] YID_scan_in,

    // Master GIN <-> PE
    input [NUMS_XBUS * NUMS_XPE - 1:0] PE_ready,
    output logic [NUMS_XBUS * NUMS_XPE - 1:0] PE_valid,
    output logic [NUMS_XBUS * NUMS_XPE * DATA_SIZE - 1:0] PE_data
);

logic [NUMS_XBUS - 1:0] XBus_ready;
logic [NUMS_XBUS - 1:0] XBus_valid;
logic [NUMS_XBUS*DATA_SIZE - 1:0] XBus_data;

/* verilator lint_off UNOPTFLAT */
logic [XID_SIZE - 1:0] scan_chain [0:NUMS_XBUS];
/* verilator lint_on UNOPTFLAT */
assign scan_chain[NUMS_XBUS] = XID_scan_in;

// Y BUS
GIN_Bus #(
    .NUMS_SLAVE(NUMS_XBUS),
    .TAG_SIZE(YID_SIZE),
    .DATA_SIZE(DATA_SIZE)
) YBus (
    .clk(clk),
    .rst(rst),
    .tag(tag_Y),
    // GLB
    .master_valid(GIN_valid),
    .master_data(GIN_data),
    .master_ready(GIN_ready),
    // Bus
    .slave_ready(XBus_ready),
    .slave_valid(XBus_valid),
    .slave_data(XBus_data),
    // Config
    .set_id(set_YID),
    .ID_scan_in(YID_scan_in),
    .ID_scan_out()
);

genvar i;
// X BUS
generate
for (i = 0; i < NUMS_XBUS; i++) begin : GIN_XBUS
    GIN_Bus #(
        .NUMS_SLAVE(NUMS_XPE),
        .TAG_SIZE(XID_SIZE),
        .DATA_SIZE(DATA_SIZE)
    ) XBus (
        .clk(clk),
        .rst(rst),
        .tag(tag_X),
        // Bus
        .master_valid(XBus_valid[i]),
        .master_data(XBus_data[(i+1)*DATA_SIZE-1 : i*DATA_SIZE]),
        .master_ready(XBus_ready[i]),
        // PE
        .slave_ready(PE_ready[(i+1)*NUMS_XPE-1 : i*NUMS_XPE]),
        .slave_valid(PE_valid[(i+1)*NUMS_XPE-1 : i*NUMS_XPE]),
        .slave_data(PE_data[(i+1)*NUMS_XPE*DATA_SIZE-1 : i*NUMS_XPE*DATA_SIZE]),
        // Config
        .set_id(set_XID),
        .ID_scan_in(scan_chain[i+1]),
        .ID_scan_out(scan_chain[i])
    );
end
endgenerate

endmodule

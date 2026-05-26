// `include "src/PE_array/GON/GON_Bus.sv"

module GON #(
    parameter   NUMS_XBUS = 6,
                NUMS_PE = 8,
                XID_SIZE = 4,
                YID_SIZE = 3,
                DATA_SIZE = 32
) (
    input clk,
    input rst,

    /* Master GON <-> GLB */
    output logic GON_valid,
    input GON_ready,
    output logic [DATA_SIZE-1:0] GON_data,

    /* Controller <-> GON */
    input [XID_SIZE-1:0] tag_X,
    input [YID_SIZE-1:0] tag_Y,
    /* config */
    input set_XID,
    input [XID_SIZE - 1:0] XID_scan_in,

    input set_YID,
    input [YID_SIZE - 1:0] YID_scan_in,

    // Master PE <-> GON
    input [NUMS_XBUS * NUMS_PE - 1:0] PE_valid,
    output logic [NUMS_XBUS * NUMS_PE - 1:0] PE_ready,
    input [DATA_SIZE * NUMS_XBUS * NUMS_PE - 1:0] PE_data

);

genvar i;

logic [NUMS_XBUS - 1:0] XBus_valid;
logic [NUMS_XBUS - 1:0] XBus_ready;
logic [NUMS_XBUS * DATA_SIZE - 1:0] XBus_data;
/* verilator lint_off UNOPTFLAT */
logic [XID_SIZE - 1:0] XID_scan_chain [0:NUMS_XBUS];
/* verilator lint_on UNOPTFLAT */

GON_Bus #(
    .NUMS_MASTER(NUMS_XBUS),
    .TAG_SIZE(YID_SIZE),
    .DATA_SIZE(DATA_SIZE)
) Y_Bus (
    .clk(clk),
    .rst(rst),
    .tag(tag_Y),
    // Bus
    .master_valid(XBus_valid),
    .master_data(XBus_data),
    .master_ready(XBus_ready),
    // GLB
    .slave_valid(GON_valid),
    .slave_ready(GON_ready),
    .slave_data(GON_data),
    .set_id(set_YID),
    .ID_scan_in(YID_scan_in),
    .ID_scan_out()
);

assign XID_scan_chain[NUMS_XBUS] = XID_scan_in;

generate
    for (i = 0; i < NUMS_XBUS; i++) begin : GON_XBUS
        GON_Bus #(
            .NUMS_MASTER(NUMS_PE),
            .TAG_SIZE(XID_SIZE),
            .DATA_SIZE(DATA_SIZE)
        ) XBus (
            .clk(clk),
            .rst(rst),
            .tag(tag_X),
            // PE
            .master_valid(PE_valid[(i+1)*NUMS_PE-1:i*NUMS_PE]),
            .master_data(PE_data[(i+1)*NUMS_PE*DATA_SIZE-1:i*NUMS_PE*DATA_SIZE]),
            .master_ready(PE_ready[(i+1)*NUMS_PE-1:i*NUMS_PE]),
            // Bus
            .slave_valid(XBus_valid[i]),
            .slave_ready(XBus_ready[i]),
            .slave_data(XBus_data[(i+1)*DATA_SIZE-1:i*DATA_SIZE]),
            .set_id(set_XID),
            .ID_scan_in(XID_scan_chain[i+1]),
            .ID_scan_out(XID_scan_chain[i])
        );
    end
endgenerate

endmodule

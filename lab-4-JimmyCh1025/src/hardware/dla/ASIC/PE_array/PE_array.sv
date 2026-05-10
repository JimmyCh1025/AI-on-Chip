// `include "src/PE_array/PE.sv"
// `include "src/PE_array/GIN/GIN.sv"
// `include "src/PE_array/GON/GON.sv"

module PE_array #(
    parameter NUMS_PE_ROW = 6,
    parameter NUMS_PE_COL = 8,
    parameter XID_SIZE = 4,
    parameter YID_SIZE = 3,
    parameter DATA_SIZE = 32,
    parameter CONFIG_SIZE = 11
)(
    input clk,
    input rst,

    /* Scan Chain */
    input set_XID,
    input [XID_SIZE-1:0] ifmap_XID_scan_in,
    input [XID_SIZE-1:0] filter_XID_scan_in,
    input [XID_SIZE-1:0] ipsum_XID_scan_in,
    input [XID_SIZE-1:0] opsum_XID_scan_in,
    // output [XID_SIZE-1:0] XID_scan_out,

    input set_YID,
    input [YID_SIZE-1:0] ifmap_YID_scan_in,
    input [YID_SIZE-1:0] filter_YID_scan_in,
    input [YID_SIZE-1:0] ipsum_YID_scan_in,
    input [YID_SIZE-1:0] opsum_YID_scan_in,
    // output logic [YID_SIZE-1:0] YID_scan_out,

    input set_LN,
    input [NUMS_PE_ROW-2:0] LN_config_in,

    /* Controller */
    input [NUMS_PE_ROW*NUMS_PE_COL-1:0] PE_en,
    input [CONFIG_SIZE-1:0] PE_config,
    input [XID_SIZE-1:0] ifmap_tag_X,
    input [YID_SIZE-1:0] ifmap_tag_Y,
    input [XID_SIZE-1:0] filter_tag_X,
    input [YID_SIZE-1:0] filter_tag_Y,
    input [XID_SIZE-1:0] ipsum_tag_X,
    input [YID_SIZE-1:0] ipsum_tag_Y,
    input [XID_SIZE-1:0] opsum_tag_X,
    input [YID_SIZE-1:0] opsum_tag_Y,

    /* GLB */
    input GLB_ifmap_valid,
    output logic GLB_ifmap_ready,
    input GLB_filter_valid,
    output logic GLB_filter_ready,
    input GLB_ipsum_valid,
    output logic GLB_ipsum_ready,
    input [DATA_SIZE-1:0] GLB_data_in,

    output logic GLB_opsum_valid,
    input GLB_opsum_ready,
    output logic [DATA_SIZE-1:0] GLB_data_out

);

genvar i, j;

logic [NUMS_PE_ROW*NUMS_PE_COL-1:0] PE_ifmap_valid;
logic [NUMS_PE_ROW*NUMS_PE_COL-1:0] PE_ifmap_ready;
logic [NUMS_PE_ROW*NUMS_PE_COL*DATA_SIZE-1:0] PE_ifmap;

logic [NUMS_PE_ROW*NUMS_PE_COL-1:0] PE_filter_valid;
logic [NUMS_PE_ROW*NUMS_PE_COL-1:0] PE_filter_ready;
logic [NUMS_PE_ROW*NUMS_PE_COL*DATA_SIZE-1:0] PE_filter;

logic [NUMS_PE_ROW*NUMS_PE_COL-1:0] GIN_ipsum_valid;
logic [NUMS_PE_ROW*NUMS_PE_COL-1:0] GIN_ipsum_ready;
logic [NUMS_PE_ROW*NUMS_PE_COL*DATA_SIZE-1:0] GIN_ipsum;

logic [NUMS_PE_ROW*NUMS_PE_COL-1:0] GON_opsum_valid;
logic [NUMS_PE_ROW*NUMS_PE_COL-1:0] GON_opsum_ready;
logic [NUMS_PE_ROW*NUMS_PE_COL*DATA_SIZE-1:0] GON_opsum;

logic [NUMS_PE_ROW*NUMS_PE_COL-1:0] LN_ipsum_valid;
logic [NUMS_PE_ROW*NUMS_PE_COL-1:0] LN_ipsum_ready;
logic [NUMS_PE_ROW*NUMS_PE_COL*DATA_SIZE-1:0] LN_ipsum;

logic [NUMS_PE_ROW*NUMS_PE_COL-1:0] PE_ipsum_valid;
logic [NUMS_PE_ROW*NUMS_PE_COL-1:0] PE_ipsum_ready;
logic [NUMS_PE_ROW*NUMS_PE_COL*DATA_SIZE-1:0] PE_ipsum;

logic [NUMS_PE_ROW*NUMS_PE_COL-1:0] PE_opsum_valid;
logic [NUMS_PE_ROW*NUMS_PE_COL-1:0] PE_opsum_ready;
logic [NUMS_PE_ROW*NUMS_PE_COL*DATA_SIZE-1:0] PE_opsum;

logic [NUMS_PE_ROW-2:0] LN_config;


always_ff @( posedge clk ) begin //LN_config
    if (rst) LN_config <= 'd0;
    else begin
        if (set_LN) LN_config <= LN_config_in;
    end
end

GIN #(
    .NUMS_XBUS(NUMS_PE_ROW),
    .NUMS_XPE(NUMS_PE_COL),
    .XID_SIZE(XID_SIZE),
    .YID_SIZE(YID_SIZE),
    .DATA_SIZE(DATA_SIZE)
) GIN_IFMAP (
    .clk(clk),
    .rst(rst),
    // Slave GLB <-> GIN
    .GIN_valid(GLB_ifmap_valid),
    .GIN_ready(GLB_ifmap_ready),
    .tag_X(ifmap_tag_X),
    .tag_Y(ifmap_tag_Y),
    .GIN_data(GLB_data_in),
    // Controller <-> GIN
    .set_XID(set_XID),
    .XID_scan_in(ifmap_XID_scan_in),
    // .XID_scan_out(XID_scan_out),
    .set_YID(set_YID),
    .YID_scan_in(ifmap_YID_scan_in),
    // .YID_scan_out(YID_scan_out),
    // Master GIN <-> PE
    .PE_valid(PE_ifmap_valid),
    .PE_ready(PE_ifmap_ready),
    .PE_data(PE_ifmap)
);

GIN #(
    .NUMS_XBUS(NUMS_PE_ROW),
    .NUMS_XPE(NUMS_PE_COL),
    .XID_SIZE(XID_SIZE),
    .YID_SIZE(YID_SIZE),
    .DATA_SIZE(DATA_SIZE)
) GIN_FILTER (
    .clk(clk),
    .rst(rst),
    // Slave GLB <-> GIN
    .GIN_valid(GLB_filter_valid),
    .GIN_ready(GLB_filter_ready),
    .tag_X(filter_tag_X),
    .tag_Y(filter_tag_Y),
    .GIN_data(GLB_data_in),
    // Controller <-> GIN
    .set_XID(set_XID),
    .XID_scan_in(filter_XID_scan_in),
    // .XID_scan_out(XID_scan_out),
    .set_YID(set_YID),
    .YID_scan_in(filter_YID_scan_in),
    // .YID_scan_out(YID_scan_out),
    // Master GIN <-> PE
    .PE_valid(PE_filter_valid),
    .PE_ready(PE_filter_ready),
    .PE_data(PE_filter)
);

GIN #(
    .NUMS_XBUS(NUMS_PE_ROW),
    .NUMS_XPE(NUMS_PE_COL),
    .XID_SIZE(XID_SIZE),
    .YID_SIZE(YID_SIZE),
    .DATA_SIZE(DATA_SIZE)
) GIN_IPSUM (
    .clk(clk),
    .rst(rst),
    // Slave GLB <-> GIN
    .GIN_valid(GLB_ipsum_valid),
    .GIN_ready(GLB_ipsum_ready),
    .tag_X(ipsum_tag_X),
    .tag_Y(ipsum_tag_Y),
    .GIN_data(GLB_data_in),
    // Controller <-> GIN
    .set_XID(set_XID),
    .XID_scan_in(ipsum_XID_scan_in),
    // .XID_scan_out(XID_scan_out),
    .set_YID(set_YID),
    .YID_scan_in(ipsum_YID_scan_in),
    // .YID_scan_out(YID_scan_out),
    // Master GIN <-> PE
    .PE_valid(GIN_ipsum_valid),
    .PE_ready(GIN_ipsum_ready),
    .PE_data(GIN_ipsum)
);

GON #(
    .NUMS_XBUS(NUMS_PE_ROW),
    .NUMS_PE(NUMS_PE_COL),
    .XID_SIZE(XID_SIZE),
    .YID_SIZE(YID_SIZE),
    .DATA_SIZE(DATA_SIZE)
) GON (
    .clk(clk),
    .rst(rst),
    .GON_valid(GLB_opsum_valid),
    .GON_ready(GLB_opsum_ready),
    .GON_data(GLB_data_out),
    .tag_X(opsum_tag_X),
    .tag_Y(opsum_tag_Y),
    .set_XID(set_XID),
    .XID_scan_in(opsum_XID_scan_in),
    .set_YID(set_YID),
    .YID_scan_in(opsum_YID_scan_in),
    .PE_valid(GON_opsum_valid),
    .PE_ready(GON_opsum_ready),
    .PE_data(GON_opsum)
);

assign PE_ipsum_valid[NUMS_PE_COL-1:0] = GIN_ipsum_valid[NUMS_PE_COL-1:0];
assign GIN_ipsum_ready[NUMS_PE_COL-1:0] = PE_ipsum_ready[NUMS_PE_COL-1:0];
assign PE_ipsum[NUMS_PE_COL*DATA_SIZE-1:0] = GIN_ipsum[NUMS_PE_COL*DATA_SIZE-1:0];

generate // LN_connect
    for (i = 1; i < NUMS_PE_ROW; i++) begin
        always_comb begin
            if (LN_config[i-1]) begin
                PE_ipsum[((i+1)*NUMS_PE_COL*DATA_SIZE-1):(i*NUMS_PE_COL*DATA_SIZE)] = PE_opsum[(i*NUMS_PE_COL*DATA_SIZE-1):((i-1)*NUMS_PE_COL*DATA_SIZE)];
                PE_ipsum_valid[(i+1)*NUMS_PE_COL-1:i*NUMS_PE_COL] = PE_opsum_valid[i*NUMS_PE_COL-1: (i-1)*NUMS_PE_COL];
                PE_opsum_ready[i*NUMS_PE_COL-1: (i-1)*NUMS_PE_COL] = PE_ipsum_ready[(i+1)*NUMS_PE_COL-1:i*NUMS_PE_COL];
            end
            else begin
                PE_ipsum[((i+1)*NUMS_PE_COL*DATA_SIZE-1):(i*NUMS_PE_COL*DATA_SIZE)] = GIN_ipsum[((i+1)*NUMS_PE_COL*DATA_SIZE-1):(i*NUMS_PE_COL*DATA_SIZE)];
                PE_ipsum_valid[(i+1)*NUMS_PE_COL-1:i*NUMS_PE_COL] = GIN_ipsum_valid[(i+1)*NUMS_PE_COL-1:i*NUMS_PE_COL];
                PE_opsum_ready[i*NUMS_PE_COL-1: (i-1)*NUMS_PE_COL] = GON_opsum_ready[i*NUMS_PE_COL-1:(i-1)*NUMS_PE_COL];
            end
        end
    end
endgenerate

generate // GIN_connect
     for (i = 1; i < NUMS_PE_ROW; i++) begin
        always_comb begin
            if (LN_config[i-1]) GIN_ipsum_ready[(i+1)*NUMS_PE_COL-1:i*NUMS_PE_COL] = 'd0;
            else GIN_ipsum_ready[(i+1)*NUMS_PE_COL-1:i*NUMS_PE_COL] = PE_ipsum_ready[(i+1)*NUMS_PE_COL-1:i*NUMS_PE_COL];
        end
     end
endgenerate

generate // GON_connect
    for (i = 1; i < NUMS_PE_ROW; i++) begin
        always_comb begin
            if (LN_config[i-1]) begin
                GON_opsum_valid[i*NUMS_PE_COL-1:(i-1)*NUMS_PE_COL] = 'd0;
                GON_opsum[i*NUMS_PE_COL*DATA_SIZE-1:(i-1)*NUMS_PE_COL*DATA_SIZE] = 'd0;
            end
            else begin
                GON_opsum_valid[i*NUMS_PE_COL-1:(i-1)*NUMS_PE_COL] = PE_opsum_valid[i*NUMS_PE_COL-1:(i-1)*NUMS_PE_COL];
                GON_opsum[i*NUMS_PE_COL*DATA_SIZE-1:(i-1)*NUMS_PE_COL*DATA_SIZE] = PE_opsum[i*NUMS_PE_COL*DATA_SIZE-1:(i-1)*NUMS_PE_COL*DATA_SIZE];
            end
        end
    end
endgenerate

assign GON_opsum_valid[NUMS_PE_ROW*NUMS_PE_COL-1:(NUMS_PE_ROW-1)*NUMS_PE_COL] = PE_opsum_valid[NUMS_PE_ROW*NUMS_PE_COL-1:(NUMS_PE_ROW-1)*NUMS_PE_COL];
assign PE_opsum_ready[NUMS_PE_ROW*NUMS_PE_COL-1:(NUMS_PE_ROW-1)*NUMS_PE_COL] = GON_opsum_ready[NUMS_PE_ROW*NUMS_PE_COL-1:(NUMS_PE_ROW-1)*NUMS_PE_COL];
assign GON_opsum[NUMS_PE_ROW*NUMS_PE_COL*DATA_SIZE-1:(NUMS_PE_ROW-1)*NUMS_PE_COL*DATA_SIZE] = PE_opsum[NUMS_PE_ROW*NUMS_PE_COL*DATA_SIZE-1:(NUMS_PE_ROW-1)*NUMS_PE_COL*DATA_SIZE];

generate
    for (i = 0; i < NUMS_PE_ROW; i++) begin : PE_ROW
        for (j = 0; j < NUMS_PE_COL; j++) begin : PE_COL
            PE PE (
                .clk(clk),
                .rst(rst),
                .PE_en(PE_en[i*NUMS_PE_COL+j]),
                .i_config(PE_config),
                .ifmap(PE_ifmap[i*NUMS_PE_COL*DATA_SIZE+(j+1)*DATA_SIZE-1:i*NUMS_PE_COL*DATA_SIZE+j*DATA_SIZE]),
                .filter(PE_filter[i*NUMS_PE_COL*DATA_SIZE+(j+1)*DATA_SIZE-1:i*NUMS_PE_COL*DATA_SIZE+j*DATA_SIZE]),
                .ipsum(PE_ipsum[i*NUMS_PE_COL*DATA_SIZE+(j+1)*DATA_SIZE-1:i*NUMS_PE_COL*DATA_SIZE+j*DATA_SIZE]),
                .ifmap_valid(PE_ifmap_valid[i*NUMS_PE_COL+j]),
                .ifmap_ready(PE_ifmap_ready[i*NUMS_PE_COL+j]),
                .filter_valid(PE_filter_valid[i*NUMS_PE_COL+j]),
                .filter_ready(PE_filter_ready[i*NUMS_PE_COL+j]),
                .ipsum_valid(PE_ipsum_valid[i*NUMS_PE_COL+j]),
                .ipsum_ready(PE_ipsum_ready[i*NUMS_PE_COL+j]),
                .opsum(PE_opsum[(i*NUMS_PE_COL+j+1)*DATA_SIZE-1:(i*NUMS_PE_COL+j)*DATA_SIZE]),
                .opsum_valid(PE_opsum_valid[i*NUMS_PE_COL+j]),
                .opsum_ready(PE_opsum_ready[i*NUMS_PE_COL+j])
            );
        end
    end
endgenerate

endmodule

`include "src/PE_array/PE.sv"
`include "src/PE_array/GIN/GIN.sv"
`include "src/PE_array/GON/GON.sv"

module PE_array #(
    parameter NUMS_PE_ROW = `NUMS_PE_ROW,
    parameter NUMS_PE_COL = `NUMS_PE_COL,
    parameter XID_BITS = `XID_BITS,
    parameter YID_BITS = `YID_BITS,
    parameter DATA_SIZE = `DATA_BITS,
    parameter CONFIG_SIZE = `CONFIG_SIZE
)(
    input clk,
    input rst,

    /* Scan Chain */
    input set_XID,
    input [`XID_BITS-1:0] ifmap_XID_scan_in,
    input [`XID_BITS-1:0] filter_XID_scan_in,
    input [`XID_BITS-1:0] ipsum_XID_scan_in,
    input [`XID_BITS-1:0] opsum_XID_scan_in,
    // output [XID_BITS-1:0] XID_scan_out,

    input set_YID,
    input [`YID_BITS-1:0] ifmap_YID_scan_in,
    input [`YID_BITS-1:0] filter_YID_scan_in,
    input [`YID_BITS-1:0] ipsum_YID_scan_in,
    input [`YID_BITS-1:0] opsum_YID_scan_in,
    // output logic [YID_BITS-1:0] YID_scan_out,

    input set_LN,
    input [`NUMS_PE_ROW-2:0] LN_config_in,

    /* Controller */
    input [`NUMS_PE_ROW*`NUMS_PE_COL-1:0] PE_en,
    input [`CONFIG_SIZE-1:0] PE_config,
    input [`XID_BITS-1:0] ifmap_tag_X,
    input [`YID_BITS-1:0] ifmap_tag_Y,
    input [`XID_BITS-1:0] filter_tag_X,
    input [`YID_BITS-1:0] filter_tag_Y,
    input [`XID_BITS-1:0] ipsum_tag_X,
    input [`YID_BITS-1:0] ipsum_tag_Y,
    input [`XID_BITS-1:0] opsum_tag_X,
    input [`YID_BITS-1:0] opsum_tag_Y,

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
    /*
                        ||          ----  ----  ----  ----       ----  ----  ---- 
                        ||         | PE || PE || PE || PE |     | PE || PE || PE |
     -------            ||          ----  ----  ----  ----       ----  ----  ---- 
    |  GLB  |           ||         | MC || MC || MC || MC |     | MC || MC || MC |
     -------            ||          ----  ----  ----  ----       ----  ----  ----
       |      /|        ||    ____   ||    ||    ||    ||   ...   ||    ||    || 
       |____ | |<------ || = | MC | ================================================ X-Bus1
             | |        ||    ----  
              \|<----   ||          ----  ----  ----  ----       ----  ----  ---- 
                     |  ||         | PE || PE || PE || PE |     | PE || PE || PE |
     ------------    |  ||     .    ----  ----  ----  ----       ----  ----  ---- 
    | Post       |   |  ||     .   | MC || MC || MC || MC |     | MC || MC || MC |
    | Processing | --   ||     .    ----  ----  ----  ----       ----  ----  ----
    | Unit       |      ||    ____   ||    ||    ||    ||   ...   ||    ||    || 
     ------------       || = | MC | ================================================ X-Bus12
                        ||    ----  
                       Y-Bus
    */

    /* Local Network
      ----   ----   ----   ----       ----   ----   ----   ----
     | PE | | PE | | PE | | PE |     | PE | | PE | | PE | | PE |
      ----   ----   ----   ----       ----   ----   ----   ---- 
      ⬆ <- LN
      ----   ----   ----   ----       ----   ----   ----   ----
     | PE | | PE | | PE | | PE |     | PE | | PE | | PE | | PE |
      ----   ----   ----   ----       ----   ----   ----   ---- 
      ⬆ <- LN
      ----   ----   ----   ----       ----   ----   ----   ----
     | PE | | PE | | PE | | PE |     | PE | | PE | | PE | | PE |
      ----   ----   ----   ----       ----   ----   ----   ---- 
    */

    logic [`DATA_BITS - 1 : 0] ifmap;
    logic [NUMS_PE_ROW * NUMS_PE_COL - 1 : 0] ifmap_valid;
    logic [NUMS_PE_ROW * NUMS_PE_COL - 1 : 0] ifmap_ready;

    logic [`DATA_BITS - 1 : 0] filter;
    logic [NUMS_PE_ROW * NUMS_PE_COL - 1 : 0] filter_valid;
    logic [NUMS_PE_ROW * NUMS_PE_COL - 1 : 0] filter_ready;

    logic [DATA_SIZE * NUMS_PE_ROW * NUMS_PE_COL - 1 : 0] ipsum;
    logic [NUMS_PE_ROW * NUMS_PE_COL - 1 : 0] ipsum_valid;
    logic [NUMS_PE_ROW * NUMS_PE_COL - 1 : 0] ipsum_ready;
    logic [DATA_SIZE - 1 : 0] ipsum_GIN;
    logic [NUMS_PE_ROW * NUMS_PE_COL - 1 : 0] ipsum_valid_GIN;

    logic [DATA_SIZE * NUMS_PE_ROW * NUMS_PE_COL - 1 : 0] opsum;
    logic [NUMS_PE_ROW * NUMS_PE_COL - 1 : 0] opsum_valid;
    logic [NUMS_PE_ROW * NUMS_PE_COL - 1 : 0] opsum_ready;
    logic [NUMS_PE_ROW * NUMS_PE_COL - 1 : 0] opsum_ready_GON;

    logic [`NUMS_PE_ROW - 2 : 0] LN_config;

    /* Global Input Network ifmap */
    GIN gin_ifmap(
        .clk        (clk),
        .rst        (rst),
        .GIN_valid  (GLB_ifmap_valid),
        .GIN_ready  (GLB_ifmap_ready),
        .GIN_data   (GLB_data_in),
        .tag_X      (ifmap_tag_X),
        .tag_Y      (ifmap_tag_Y),
        .set_XID    (set_XID),
        .XID_scan_in(ifmap_XID_scan_in),
        .set_YID    (set_YID),
        .YID_scan_in(ifmap_YID_scan_in),
        .PE_ready   (ifmap_ready),
        .PE_valid   (ifmap_valid),
        .PE_data    (ifmap)
    );

    /* Global Input Network filter */
    GIN gin_filter(
        .clk        (clk),
        .rst        (rst),
        .GIN_valid  (GLB_filter_valid),
        .GIN_ready  (GLB_filter_ready),
        .GIN_data   (GLB_data_in),
        .tag_X      (filter_tag_X),
        .tag_Y      (filter_tag_Y),
        .set_XID    (set_XID),
        .XID_scan_in(filter_XID_scan_in),
        .set_YID    (set_YID),
        .YID_scan_in(filter_YID_scan_in),
        .PE_ready   (filter_ready),
        .PE_valid   (filter_valid),
        .PE_data    (filter)
    );

    /* Global Input Network ipsum */
    GIN gin_ipsum(
        .clk        (clk),
        .rst        (rst),
        .GIN_valid  (GLB_ipsum_valid),
        .GIN_ready  (GLB_ipsum_ready),
        .GIN_data   (GLB_data_in),
        .tag_X      (ipsum_tag_X),
        .tag_Y      (ipsum_tag_Y),
        .set_XID    (set_XID),
        .XID_scan_in(ipsum_XID_scan_in),
        .set_YID    (set_YID),
        .YID_scan_in(ipsum_YID_scan_in),
        .PE_ready   (ipsum_ready),
        .PE_valid   (ipsum_valid_GIN),
        .PE_data    (ipsum_GIN)
    );



    /* Processing Element */
    generate 
        for (genvar i = 0 ; i < NUMS_PE_ROW ; ++i) begin : pe_row
            for (genvar j = 0 ; j < NUMS_PE_COL ; ++j) begin : pe_col
                PE pe(
                    .clk         (clk),
                    .rst         (rst),
                    .PE_en       (PE_en[i * NUMS_PE_COL + j]),                              // input
                    .i_config    (PE_config),                                               // input
                    .ifmap       (ifmap),                                                   // input
                    .filter      (filter),                                                  // input
                    .ipsum       (ipsum[((i * NUMS_PE_COL + j) * DATA_SIZE) +: DATA_SIZE]), // input
                    .ifmap_valid (ifmap_valid[i * NUMS_PE_COL + j]),                        // input
                    .filter_valid(filter_valid[i * NUMS_PE_COL + j]),                       // input
                    .ipsum_valid (ipsum_valid[i * NUMS_PE_COL + j]),                        // input
                    .opsum_ready (opsum_ready[i * NUMS_PE_COL + j]),                        // input
                    .opsum       (opsum[((i * NUMS_PE_COL + j) * DATA_SIZE) +: DATA_SIZE]), // output
                    .ifmap_ready (ifmap_ready[i * NUMS_PE_COL + j]),                        // output
                    .filter_ready(filter_ready[i * NUMS_PE_COL + j]),                       // output
                    .ipsum_ready (ipsum_ready[i * NUMS_PE_COL + j]),                        // output
                    .opsum_valid (opsum_valid[i * NUMS_PE_COL + j])                         // output
                );
            end
        end
    endgenerate

    /* Global Output Network opsum */
    GON gon_opsum(
        .clk        (clk),  
        .rst        (rst),
        .GON_valid  (GLB_opsum_valid),   // output
        .GON_ready  (GLB_opsum_ready),   // input
        .GON_data   (GLB_data_out),      // output
        .tag_X      (opsum_tag_X),       // input
        .tag_Y      (opsum_tag_Y),       // input
        .set_XID    (set_XID),           // input
        .XID_scan_in(opsum_XID_scan_in), // input
        .set_YID    (set_YID),           // input
        .YID_scan_in(opsum_YID_scan_in), // input
        .PE_valid   (opsum_valid),       // input
        .PE_ready   (opsum_ready_GON),   // output
        .PE_data    (opsum)              // input
    );

    /* Local Network */
    always_ff @(posedge clk) begin
        if (set_LN) begin
            LN_config <= LN_config_in;
        end else begin
            LN_config <= LN_config;
        end
    end

    always_comb begin
        /*
            PE
            ⬆ <- LN_config valid => opsum
            PE
            ⬆ <- LN_config invalid => glb data
            PE 
            :
            PE <- bottom get data from glb
        */

        // ipsum 
        for (int i = 0 ; i < NUMS_PE_ROW - 1 ; ++i) begin
            for (int j = 0 ; j < NUMS_PE_COL ; ++j) begin
                // if LN_config is true, then get data from opsum, else get data from glb 
                if (LN_config[i]) begin
                    ipsum[((i * NUMS_PE_COL + j) * DATA_SIZE) +: DATA_SIZE] = opsum[(((i+1) * NUMS_PE_COL + j) * DATA_SIZE) +: DATA_SIZE];
                    ipsum_valid[i * NUMS_PE_COL + j] = opsum_valid[(i + 1) * NUMS_PE_COL + j];
                end else begin
                    ipsum[((i * NUMS_PE_COL + j) * DATA_SIZE) +: DATA_SIZE] = GLB_data_in;
                    ipsum_valid[i * NUMS_PE_COL + j] = ipsum_valid_GIN[i * NUMS_PE_COL + j];
                end
            end
        end

        // bottom : Get Global Buffer(GLB) data 
        for (int i = 0 ; i < NUMS_PE_COL ; ++i) begin
            ipsum[(((NUMS_PE_ROW - 1) * NUMS_PE_COL + i) * DATA_SIZE) +: DATA_SIZE] = GLB_data_in;
            ipsum_valid[(NUMS_PE_ROW - 1) * NUMS_PE_COL + i] = ipsum_valid_GIN[(NUMS_PE_ROW - 1) * NUMS_PE_COL + i];
        end

        for (int i = 1 ; i < NUMS_PE_ROW ; ++i) begin
            for (int j = 0; j < NUMS_PE_COL ; ++j) begin
                // valid => ready 
                if (LN_config[i-1]) begin
                    opsum_ready[i * NUMS_PE_COL + j] = ipsum_ready[(i - 1) * NUMS_PE_COL + j]; 
                end else begin
                    opsum_ready[i * NUMS_PE_COL + j] = opsum_ready_GON[i * NUMS_PE_COL + j];
                end
            end
        end
        
        opsum_ready[NUMS_PE_COL-1:0] = opsum_ready_GON[NUMS_PE_COL-1:0];
    end

    

endmodule
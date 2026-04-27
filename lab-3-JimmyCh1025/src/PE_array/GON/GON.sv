`include "src/PE_array/GON/GON_Bus.sv"
`include "src/PE_array/GON/GON_MulticastController.sv"

module GON (
    input clk,
    input rst,

    /* Master GON <-> GLB */
    output logic GON_valid,
    input GON_ready,
    output logic [`DATA_BITS-1:0] GON_data,

    /* Controller <-> GON */
    input [`XID_BITS-1:0] tag_X,
    input [`YID_BITS-1:0] tag_Y,
    /* config */
    input set_XID,
    input [`XID_BITS - 1:0] XID_scan_in,

    input set_YID,
    input [`YID_BITS - 1:0] YID_scan_in,

    // Master PE <-> GON
    input [`NUMS_PE_ROW * `NUMS_PE_COL - 1:0] PE_valid,
    output logic [`NUMS_PE_ROW * `NUMS_PE_COL - 1:0] PE_ready,
    input [`DATA_BITS * `NUMS_PE_ROW * `NUMS_PE_COL - 1:0] PE_data

);
    /*
                    ____  ____  ____  ____       ____  ____  ____ 
                   | MC || MC || MC || MC |     | MC || MC || MC |
        ||          ----  ----  ----  ----       ----  ----  ----
        ||    ____   ||    ||    ||    ||   ...   ||    ||    || 
        || = | MC | ================================================ X-Bus1
        ||    ----
        ||     .    ____  ____  ____  ____       ____  ____  ____ 
        ||     .   | MC || MC || MC || MC |     | MC || MC || MC |
        ||     .    ----  ----  ----  ----       ----  ----  ----
        ||    ____   ||    ||    ||    ||   ...   ||    ||    || 
        || = | MC | ================================================ X-Bus12
        ||    ----  
       Y-Bus
    */

    /* 
        slave                                                master
         -----  Y_Bus_valid      X_Bus_valid        PE_valid
        | GLB |     <-  | Y_Bus |    <-   | X_Bus |    <-    | PE |
         -----      ->               ->                -> 
                GLB_ready        Y_Bus_ready        X_Bus_ready
    */
    
    logic [`NUMS_PE_ROW-1 : 0] X_Bus_valid;
    logic [`NUMS_PE_ROW-1 : 0] X_Bus_ready;

    logic [`XID_BITS-1:0] MC_id [`NUMS_PE_ROW-1:0];
    logic [`XID_BITS-1:0] MC_id_out [`NUMS_PE_ROW-1:0];

    GON_Bus #(
        .NUMS_MASTER (`NUMS_PE_ROW),
        .ID_SIZE     (`YID_BITS)
    ) GON_Y_Bus (
        .clk         (clk),
        .rst         (rst),
        .tag         (tag_Y),
        .master_valid(X_Bus_valid), // input
        .master_data (),            // input
        .master_ready(X_Bus_ready), // output
        .slave_ready (GON_ready),   // input
        .slave_valid (GON_valid),   // output
        .slave_data  (),            // output
        .set_id      (set_YID),
        .ID_scan_in  (YID_scan_in),
        .ID_scan_out ()             // output
    );


    generate 
        for (genvar i = 0 ; i < `NUMS_PE_ROW ; ++i) begin : gon_bus_gen
            GON_Bus #(
                .NUMS_MASTER (`NUMS_PE_COL),
                .ID_SIZE     (`XID_BITS)
            ) GON_X_Bus (
                .clk         (clk),
                .rst         (rst),
                .tag         (tag_X),
                .master_valid(PE_valid[i * `NUMS_PE_COL +: `NUMS_PE_COL]),                            // input
                .master_data (PE_data[(i * `NUMS_PE_COL * `DATA_BITS) +: `NUMS_PE_COL * `DATA_BITS]), // input                                          // input
                .master_ready(PE_ready[i * `NUMS_PE_COL +: `NUMS_PE_COL]),                            // output
                .slave_ready (X_Bus_ready[i]),                                                        // input
                .slave_valid (X_Bus_valid[i]),                                                        // output
                .slave_data  (),                                                                      // output
                .set_id      (set_XID),
                .ID_scan_in  (MC_id[i]),
                .ID_scan_out (MC_id_out[i])
            );
        end
    endgenerate

    always_comb begin
        MC_id[0] = XID_scan_in;
        for (int i = 1 ; i < `NUMS_PE_ROW ; ++i) begin
            MC_id[i] = MC_id_out[i-1];
        end
    end

    always_comb begin
        GON_data = '0;
        for (int i = 0 ; i < `NUMS_PE_ROW * `NUMS_PE_COL ; ++i) begin
            if (PE_ready[i]) begin
                GON_data = PE_data[(i * `DATA_BITS) +: `DATA_BITS];
            end
        end
    end

endmodule

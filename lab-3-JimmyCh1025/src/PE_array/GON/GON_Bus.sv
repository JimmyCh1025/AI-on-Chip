module GON_Bus #(
    parameter NUMS_MASTER = `NUMS_PE_COL,
    parameter ID_SIZE = `XID_BITS
) (
    input clk,
    input rst,
    input [ID_SIZE - 1:0] tag,

    input [NUMS_MASTER - 1:0] master_valid,
    input [NUMS_MASTER * `DATA_BITS - 1:0] master_data,
    output logic [NUMS_MASTER - 1:0] master_ready,

    output logic slave_valid,
    input slave_ready,
    output logic [`DATA_BITS - 1:0] slave_data,

    // Config
    input set_id,
    input [ID_SIZE - 1:0] ID_scan_in,
    output logic [ID_SIZE - 1 :0] ID_scan_out
 );

    /*      
            ____  ____  ____  ____       ____  ____  ____ 
           | MC || MC || MC || MC |     | MC || MC || MC |
            ----  ----  ----  ----       ----  ----  ----
             ||    ||    ||    ||   ...   ||    ||    || 
       bus ==============================================
    */
    
    logic [ID_SIZE-1:0] MC_id [NUMS_MASTER-1:0];
    logic [ID_SIZE-1:0] MC_id_out [NUMS_MASTER-1:0];
    logic [NUMS_MASTER-1:0] master_valid_out;

    // pass ID from MC 0 to MC N
    always_comb begin
        MC_id[0] = ID_scan_in;
        for (int i = 1 ; i < NUMS_MASTER ; ++i) begin
            MC_id[i] = MC_id_out[i-1];
        end
        ID_scan_out = MC_id_out[NUMS_MASTER-1];
    end

    // generate NUMS_MASTER Multicast Controller 
    generate
        for (genvar i = 0 ; i < NUMS_MASTER ; ++i) begin : gon_mc_gen
            GON_MulticastController #(
                .ID_SIZE(ID_SIZE)
            ) GON_MC (
                .clk      (clk),
                .rst      (rst),
                .set_id   (set_id),
                .id_in    (MC_id[i]),
                .id       (MC_id_out[i]),
                .tag      (tag),
                .valid_in (master_valid[i]),
                .valid_out(master_valid_out[i]),
                .ready_in (slave_ready),
                .ready_out(master_ready[i])
            );
        end
    endgenerate

   
    always_comb begin
        // if any slave ready, then master ready is true, else false
        slave_valid = |master_valid_out;
    end

endmodule

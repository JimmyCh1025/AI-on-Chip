 module GIN_Bus #(
    parameter NUMS_SLAVE = `NUMS_PE_COL,
    parameter ID_SIZE = `XID_BITS
) (
    input clk,
    input rst,

   // Master I/O
    input [ID_SIZE-1:0] tag,
    input master_valid,
    input [`DATA_BITS-1:0] master_data,
    output logic master_ready,

   // Slave I/O
    input [NUMS_SLAVE-1:0] slave_ready,
    output logic [NUMS_SLAVE-1:0] slave_valid,
    output logic [`DATA_BITS-1:0] slave_data,

    // Config
    input set_id,
    input [ID_SIZE-1:0] ID_scan_in,
    output logic [ID_SIZE-1:0] ID_scan_out
 );
    /*     
            ____  ____  ____  ____       ____  ____  ____ 
           | MC || MC || MC || MC |     | MC || MC || MC |
            ----  ----  ----  ----       ----  ----  ----
             ||    ||    ||    ||   ...   ||    ||    || 
       bus ==============================================
    */

    logic [ID_SIZE-1:0] MC_id [NUMS_SLAVE-1:0];
    logic [ID_SIZE-1:0] MC_id_out [NUMS_SLAVE-1:0];

    // pass ID from MC 0 to MC N
    always_comb begin
        MC_id[0] = ID_scan_in;
        for (int i = 1 ; i < NUMS_SLAVE ; ++i) begin
            MC_id[i] = MC_id_out[i-1];
        end
        ID_scan_out = MC_id_out[NUMS_SLAVE-1];
    end

    // generate NUMS_SLAVE Multicast Controller 
    generate
        for (genvar i = 0 ; i < NUMS_SLAVE ; ++i) begin : gin_mc_gen
            GIN_MulticastController #(
                .ID_SIZE(ID_SIZE)
            ) GIN_MC (
                .clk      (clk),
                .rst      (rst),
                .set_id   (set_id),
                .id_in    (MC_id[i]),
                .id       (MC_id_out[i]),
                .tag      (tag),
                .valid_in (master_valid),
                .valid_out(slave_valid[i]),
                .ready_in (slave_ready[i]),
                .ready_out()                // Global input network => no output
            );
        end
    endgenerate

   
    always_comb begin
        // if master valid, then write data to slave, else 0
        slave_data = master_valid ? master_data : '0;
        // if any slave ready, then master ready is true, else false
        master_ready = |slave_ready;
    end

endmodule

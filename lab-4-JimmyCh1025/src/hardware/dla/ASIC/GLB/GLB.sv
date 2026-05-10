`include "AXI_define.svh"
`include "ASIC.svh"

module GLB (
  input clk,
  input EN,  // active low
  input WEB, // active low
  input MODE, // Word:1, Byte:0
  input [`GLB_ADDR_BITS-1:0] A, // Byte Address (64KB, 16bits)
  input [`DATA_BITS-1:0] DI,
  output logic [`DATA_BITS-1:0] DO
);

/* GLB is made of SRAM */


  /* SRAM IO */
  logic [31:0] BWEB;
  logic [31:0] SRAM_DI;
  logic [31:0] SRAM_DO;

  /* Address allignment */
  always_comb begin
    // write mask (store unit)
    if(EN) begin
      BWEB = 32'hffffffff;
    end else begin
        if(MODE == `BYTE_MODE)begin
            case (A[1:0])
                2'd0: BWEB = 32'hffffff00;
                2'd1: BWEB = 32'hffff00ff;
                2'd2: BWEB = 32'hff00ffff;
                2'd3: BWEB = 32'h00ffffff;
            endcase
        end else begin
            BWEB = 32'h00000000;
        end
    end

    /* DI allignment */
    if(MODE == `BYTE_MODE)begin
        case (A[1:0])
            2'd0: SRAM_DI = {24'd0, DI[7:0]};
            2'd1: SRAM_DI = {16'd0, DI[7:0], 8'd0};
            2'd2: SRAM_DI = {8'd0,  DI[7:0], 16'd0};
            2'd3: SRAM_DI = {DI[7:0], 24'd0};
        endcase
    end else begin
        SRAM_DI = DI;
    end

    /* DO allignment */
    if(MODE == `BYTE_MODE)begin
        case (A[1:0])
            2'd0: DO = {24'd0,SRAM_DO[7:0]};
            2'd1: DO = {24'd0,SRAM_DO[15:8]};
            2'd2: DO = {24'd0,SRAM_DO[23:16]};
            2'd3: DO = {24'd0,SRAM_DO[31:24]};
        endcase
    end else begin
        DO = SRAM_DO;
    end
  end

  SRAM i_SRAM (
    .SLP(1'b0),
    .DSLP(1'b0),
    .SD(1'b0),
    .PUDELAY(),
    .CLK(clk),
    .CEB(EN), // active low
    .WEB(WEB),
    .A(A[`GLB_ADDR_BITS-1:2]),
    .D(SRAM_DI),
    .BWEB(BWEB),
    .RTSEL(2'b01),
    .WTSEL(2'b01),
    .Q(SRAM_DO)
  );


endmodule

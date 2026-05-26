`include "AXI_define.svh"
`include "ASIC.svh"

module asic_wrapper (
    input ACLK,
    input ARESETn,

    output logic ASIC_interrupt,

    /*************** AXI slave ***************/
    //WRITE ADDRESS0
	input [`AXI_IDS_BITS-1:0] AWID_S,
	input [`AXI_ADDR_BITS-1:0] AWADDR_S,
	input [`AXI_LEN_BITS-1:0] AWLEN_S,
	input [`AXI_SIZE_BITS-1:0] AWSIZE_S,
	input [1:0] AWBURST_S,
	input AWVALID_S,
	output logic AWREADY_S,

	//WRITE DATA0
	input [`AXI_DATA_BITS-1:0] WDATA_S,
	input [`AXI_STRB_BITS-1:0] WSTRB_S,
	input WLAST_S,
	input WVALID_S,
	output logic WREADY_S,

	//WRITE RESPONSE0
	output logic [`AXI_IDS_BITS-1:0] BID_S,
	output logic [1:0] BRESP_S,
	output logic BVALID_S,
	input BREADY_S,

  //READ ADDRESS0
	input [`AXI_IDS_BITS-1:0] ARID_S,
	input [`AXI_ADDR_BITS-1:0] ARADDR_S,
	input [`AXI_LEN_BITS-1:0] ARLEN_S,
	input [`AXI_SIZE_BITS-1:0] ARSIZE_S,
	input [1:0] ARBURST_S,
	input ARVALID_S,
	output logic ARREADY_S,

	//READ DATA0
	output logic [`AXI_IDS_BITS-1:0] RID_S,
	output logic [`AXI_DATA_BITS-1:0] RDATA_S,
	output logic [1:0] RRESP_S,
	output logic RLAST_S,
	output logic RVALID_S,
	input RREADY_S,

    /*************** AXI master ***************/
    //WRITE ADDRESS0
    output logic [`AXI_ID_BITS-1:0] AWID_M,
    output logic [`AXI_ADDR_BITS-1:0] AWADDR_M,
    output logic [`AXI_LEN_BITS-1:0] AWLEN_M,
    output logic [`AXI_SIZE_BITS-1:0] AWSIZE_M,
    output logic [1:0] AWBURST_M,
    output logic AWVALID_M,
    input AWREADY_M,

    //WRITE DATA0
    output logic [`AXI_DATA_BITS-1:0] WDATA_M,
    output logic [`AXI_STRB_BITS-1:0] WSTRB_M,
    output logic WLAST_M,
    output logic WVALID_M,
    input WREADY_M,

    //WRITE RESPONSE0
    input [`AXI_ID_BITS-1:0] BID_M,
    input [1:0] BRESP_M,
    input BVALID_M,
    output logic BREADY_M,

    //READ ADDRESS0
    output logic [`AXI_ID_BITS-1:0] ARID_M,
    output logic [`AXI_ADDR_BITS-1:0] ARADDR_M,
    output logic [`AXI_LEN_BITS-1:0] ARLEN_M,
    output logic [`AXI_SIZE_BITS-1:0] ARSIZE_M,
    output logic [1:0] ARBURST_M,
    output logic ARVALID_M,
    input logic ARREADY_M,

    //READ DATA0
    input [`AXI_ID_BITS-1:0] RID_M,
    input [`AXI_DATA_BITS-1:0] RDATA_M,
    input [1:0] RRESP_M,
    input RLAST_M,
    input RVALID_M,
    output logic RREADY_M
);

/***********************************
        Global Buffer Mux
************************************/
logic GLB_EN, GLB_EN_dma, GLB_EN_asic;
logic GLB_WEB, GLB_WEB_dma, GLB_WEB_asic;
logic GLB_MODE, GLB_MODE_dma, GLB_MODE_asic;
logic [`GLB_ADDR_BITS-1:0] GLB_A, GLB_A_dma, GLB_A_asic;
logic [`DATA_BITS-1:0] GLB_DI, GLB_DI_dma, GLB_DI_asic;
logic [`DATA_BITS-1:0] GLB_DO, GLB_DO_dma, GLB_DO_asic;
logic glb_mux;

always_comb begin
    // output (mux)
    GLB_DO_dma = (glb_mux == `DMA)?GLB_DO:`DATA_BITS'd0;
    GLB_DO_asic = (glb_mux == `ASIC)?GLB_DO:`DATA_BITS'd0;

    // input (mux)
    if(glb_mux == `ASIC) begin
        GLB_EN = GLB_EN_asic;
        GLB_WEB = GLB_WEB_asic;
        GLB_MODE = GLB_MODE_asic;
        GLB_A = GLB_A_asic;
        GLB_DI = GLB_DI_asic;
    end else begin
        GLB_EN = GLB_EN_dma;
        GLB_WEB = GLB_WEB_dma;
        GLB_MODE = GLB_MODE_dma;
        GLB_A = GLB_A_dma;
        GLB_DI = GLB_DI_dma;
    end
end
/***********************************
            Global Buffer
************************************/
GLB GLB_0(
  .clk(ACLK),
  .EN(GLB_EN),
  .WEB(GLB_WEB),
  .MODE(GLB_MODE),
  .A(GLB_A),
  .DI(GLB_DI),
  .DO(GLB_DO)
);

/***********************************
            ASIC DMA
************************************/
logic DMA_EN;
logic [1:0] DMA_MODE;
logic [1:0] DMA_BYTE_BIAS;
logic DMA_DONE;
logic [`AXI_ADDR_BITS-1:0] DMA_DRAM_ADDR;
logic [`GLB_ADDR_BITS-1:0] DMA_GLB_ADDR;
logic [`GLB_ADDR_BITS-1:0] DMA_LEN;

DMA DMA_0(
    .clk(ACLK),
    .rst(~ARESETn),

    /* controller */
    .EN(DMA_EN),
    .MODE(DMA_MODE),
    .BYTE_BIAS(DMA_BYTE_BIAS),
    .DONE(DMA_DONE),
    .DRAM_ADDR(DMA_DRAM_ADDR),
    .GLB_ADDR(DMA_GLB_ADDR),
    .LEN(DMA_LEN),

    /*************** AXI master ***************/
    //WRITE ADDRESS0
    .AWID_M(AWID_M),
    .AWADDR_M(AWADDR_M),
    .AWLEN_M(AWLEN_M),
    .AWSIZE_M(AWSIZE_M),
    .AWBURST_M(AWBURST_M),
    .AWVALID_M(AWVALID_M),
    .AWREADY_M(AWREADY_M),

    //WRITE DATA0
    .WDATA_M(WDATA_M),
    .WSTRB_M(WSTRB_M),
    .WLAST_M(WLAST_M),
    .WVALID_M(WVALID_M),
    .WREADY_M(WREADY_M),

    //WRITE RESPONSE0
    .BID_M(BID_M),
    .BRESP_M(BRESP_M),
    .BVALID_M(BVALID_M),
    .BREADY_M(BREADY_M),

    //READ ADDRESS0
    .ARID_M(ARID_M),
    .ARADDR_M(ARADDR_M),
    .ARLEN_M(ARLEN_M),
    .ARSIZE_M(ARSIZE_M),
    .ARBURST_M(ARBURST_M),
    .ARVALID_M(ARVALID_M),
    .ARREADY_M(ARREADY_M),

    //READ DATA0
    .RID_M(RID_M),
    .RDATA_M(RDATA_M),
    .RRESP_M(RRESP_M),
    .RLAST_M(RLAST_M),
    .RVALID_M(RVALID_M),
    .RREADY_M(RREADY_M),

    /* GLB */
    .GLB_EN(GLB_EN_dma),
    .GLB_WEB(GLB_WEB_dma),
    .GLB_MODE(GLB_MODE_dma),
    .GLB_A(GLB_A_dma),
    .GLB_DI(GLB_DI_dma),
    .GLB_DO(GLB_DO_dma)
);

/*****************************************
        ASIC slave ( MMIO config )
*****************************************/

  /* AXI slave FSM */
  typedef enum logic [2:0] {
        IDLE,
        WRITE_ADDR,
        WRITE_DATA,
        WRITE_RESP,
        READ_ADDR,
        READ_DATA,
        DONE_S
  } AXI_state;

  AXI_state cs_slave, cs_slave_next;
  logic [`AXI_ADDR_BITS-1:0] addr_S_reg, addr_S_reg_next;
  logic [`AXI_IDS_BITS-1:0] BID_S_next, RID_S_next;
  logic write_error, write_error_next;

  logic [`DATA_BITS-1:0] ASIC_ENABLE, ASIC_ENABLE_next;
  logic [`DATA_BITS-1:0] ASIC_MAPPING_PARAM, ASIC_MAPPING_PARAM_next;
  logic [`DATA_BITS-1:0] ASIC_SHAPE_PARAM1, ASIC_SHAPE_PARAM1_next;
  logic [`DATA_BITS-1:0] ASIC_SHAPE_PARAM2, ASIC_SHAPE_PARAM2_next;
  logic [`DATA_BITS-1:0] ASIC_IFMAP_ADDR, ASIC_IFMAP_ADDR_next;
  logic [`DATA_BITS-1:0] ASIC_FILTER_ADDR, ASIC_FILTER_ADDR_next;
  logic [`DATA_BITS-1:0] ASIC_BIAS_ADDR, ASIC_BIAS_ADDR_next;
  logic [`DATA_BITS-1:0] ASIC_OPSUM_ADDR, ASIC_OPSUM_ADDR_next;
  logic [`DATA_BITS-1:0] ASIC_GLB_FILTER_ADDR, ASIC_GLB_FILTER_ADDR_next;
  logic [`DATA_BITS-1:0] ASIC_GLB_OFMAP_ADDR, ASIC_GLB_OFMAP_ADDR_next;
  logic [`DATA_BITS-1:0] ASIC_GLB_BIAS_ADDR, ASIC_GLB_BIAS_ADDR_next;
  logic [`DATA_BITS-1:0] ASIC_IFMAP_LEN, ASIC_IFMAP_LEN_next;
  logic [`DATA_BITS-1:0] ASIC_OFMAP_LEN, ASIC_OFMAP_LEN_next;

  // Sequential logic
  always_ff @(posedge ACLK) begin
    if (~ARESETn) begin
      cs_slave <= IDLE;
      addr_S_reg <= `AXI_ADDR_BITS'd0;
      BID_S <= `AXI_IDS_BITS'd0;
      RID_S <= `AXI_IDS_BITS'd0;
      write_error <= 1'b0;
      ASIC_ENABLE <= `DATA_BITS'd0;
      ASIC_MAPPING_PARAM <= `DATA_BITS'd0;
      ASIC_SHAPE_PARAM1 <= `DATA_BITS'd0;
      ASIC_SHAPE_PARAM2 <= `DATA_BITS'd0;
      ASIC_IFMAP_ADDR <= `DATA_BITS'd0;
      ASIC_FILTER_ADDR <= `DATA_BITS'd0;
      ASIC_BIAS_ADDR <= `DATA_BITS'd0;
      ASIC_OPSUM_ADDR <= `DATA_BITS'd0;
      ASIC_GLB_FILTER_ADDR <= `DATA_BITS'd0;
      ASIC_GLB_OFMAP_ADDR <= `DATA_BITS'd0;
      ASIC_GLB_BIAS_ADDR <= `DATA_BITS'd0;
      ASIC_IFMAP_LEN <= `DATA_BITS'd0;
      ASIC_OFMAP_LEN <= `DATA_BITS'd0;
    end
    else begin
      cs_slave <= cs_slave_next;
      addr_S_reg <= addr_S_reg_next;
      BID_S <= BID_S_next;
      RID_S <= RID_S_next;
      write_error <= write_error_next;
      ASIC_ENABLE <= ASIC_ENABLE_next;
      ASIC_MAPPING_PARAM <= ASIC_MAPPING_PARAM_next;
      ASIC_SHAPE_PARAM1 <= ASIC_SHAPE_PARAM1_next;
      ASIC_SHAPE_PARAM2 <= ASIC_SHAPE_PARAM2_next;
      ASIC_IFMAP_ADDR <= ASIC_IFMAP_ADDR_next;
      ASIC_FILTER_ADDR <= ASIC_FILTER_ADDR_next;
      ASIC_BIAS_ADDR <= ASIC_BIAS_ADDR_next;
      ASIC_OPSUM_ADDR <= ASIC_OPSUM_ADDR_next;
      ASIC_GLB_FILTER_ADDR <= ASIC_GLB_FILTER_ADDR_next;
      ASIC_GLB_OFMAP_ADDR <= ASIC_GLB_OFMAP_ADDR_next;
      ASIC_GLB_BIAS_ADDR <= ASIC_GLB_BIAS_ADDR_next;
      ASIC_IFMAP_LEN <= ASIC_IFMAP_LEN_next;
      ASIC_OFMAP_LEN <= ASIC_OFMAP_LEN_next;
    end
  end

  // Combinational logic
  always_comb begin
    // ASIC config
    ASIC_ENABLE_next = ASIC_ENABLE;
    ASIC_MAPPING_PARAM_next = ASIC_MAPPING_PARAM;
    ASIC_SHAPE_PARAM1_next = ASIC_SHAPE_PARAM1;
    ASIC_SHAPE_PARAM2_next = ASIC_SHAPE_PARAM2;
    ASIC_IFMAP_ADDR_next = ASIC_IFMAP_ADDR;
    ASIC_FILTER_ADDR_next = ASIC_FILTER_ADDR;
    ASIC_BIAS_ADDR_next = ASIC_BIAS_ADDR;
    ASIC_OPSUM_ADDR_next = ASIC_OPSUM_ADDR;
    ASIC_GLB_FILTER_ADDR_next = ASIC_GLB_FILTER_ADDR;
    ASIC_GLB_OFMAP_ADDR_next = ASIC_GLB_OFMAP_ADDR;
    ASIC_GLB_BIAS_ADDR_next = ASIC_GLB_BIAS_ADDR;
    ASIC_IFMAP_LEN_next = ASIC_IFMAP_LEN;
    ASIC_OFMAP_LEN_next = ASIC_OFMAP_LEN;

    // Default assignments
    cs_slave_next = cs_slave;
    addr_S_reg_next = addr_S_reg;
    BID_S_next = BID_S;
    RID_S_next = RID_S;
    write_error_next = write_error;

    // AXI interface defaults
    AWREADY_S = 1'b0;
    WREADY_S = 1'b0;
    BRESP_S = `AXI_RESP_OKAY;    // OKAY response
    BVALID_S = 1'b0;
    ARREADY_S = 1'b0;
    RDATA_S = `AXI_DATA_BITS'd0;       // Connect SRAM output to AXI read data
    RRESP_S = `AXI_RESP_OKAY;    // OKAY response
    RLAST_S = 1'b0;
    RVALID_S = 1'b0;

    case (cs_slave)
      IDLE: begin
        if (AWVALID_S) begin // got write request
          AWREADY_S = 1'b1;  // Ready to accept write address
          cs_slave_next = WRITE_DATA;
          BID_S_next = AWID_S;
          addr_S_reg_next = AWADDR_S;  // Word aligned address
        end
        else if (ARVALID_S) begin // got read request
          ARREADY_S = 1'b1;  // Ready to accept read address
          cs_slave_next = READ_DATA; // just register output
          RID_S_next = ARID_S;
          addr_S_reg_next = ARADDR_S;  // Word aligned address
        end
      end

      READ_DATA: begin
        RVALID_S = 1'b1; // data valid
        RLAST_S = 1'b1; // always the last one data

        // read data select
        case (addr_S_reg)
          `ASIC_ENABLE_OFFSET: begin
            RDATA_S = ASIC_ENABLE;
            RRESP_S = `AXI_RESP_OKAY;
          end
          `ASIC_MAPPING_PARAM_OFFSET: begin
            RDATA_S = ASIC_MAPPING_PARAM;
            RRESP_S = `AXI_RESP_OKAY;
          end
          `ASIC_SHAPE_PARAM1_OFFSET: begin
            RDATA_S = ASIC_SHAPE_PARAM1;
            RRESP_S = `AXI_RESP_OKAY;
          end
          `ASIC_SHAPE_PARAM2_OFFSET: begin
            RDATA_S = ASIC_SHAPE_PARAM2;
            RRESP_S = `AXI_RESP_OKAY;
          end
          `ASIC_IFMAP_ADDR_OFFSET: begin
            RDATA_S = ASIC_IFMAP_ADDR;
            RRESP_S = `AXI_RESP_OKAY;
          end
          `ASIC_FILTER_ADDR_OFFSET: begin
            RDATA_S = ASIC_FILTER_ADDR;
            RRESP_S = `AXI_RESP_OKAY;
          end
          `ASIC_BIAS_ADDR_OFFSET: begin
            RDATA_S = ASIC_BIAS_ADDR;
            RRESP_S = `AXI_RESP_OKAY;
          end
          `ASIC_OPSUM_ADDR_OFFSET: begin
            RDATA_S = ASIC_OPSUM_ADDR;
            RRESP_S = `AXI_RESP_OKAY;
          end
          `ASIC_GLB_FILTER_ADDR_OFFSET: begin
            RDATA_S = ASIC_GLB_FILTER_ADDR;
            RRESP_S = `AXI_RESP_OKAY;
          end
          `ASIC_GLB_OFMAP_ADDR_OFFSET: begin
            RDATA_S = ASIC_GLB_OFMAP_ADDR;
            RRESP_S = `AXI_RESP_OKAY;
          end
          `ASIC_GLB_BIAS_ADDR_OFFSET: begin
            RDATA_S = ASIC_GLB_BIAS_ADDR;
            RRESP_S = `AXI_RESP_OKAY;
          end
          `ASIC_IFMAP_LEN_OFFSET: begin
            RDATA_S = ASIC_IFMAP_LEN;
            RRESP_S = `AXI_RESP_OKAY;
          end
          `ASIC_OFMAP_LEN_OFFSET: begin
            RDATA_S = ASIC_OFMAP_LEN;
            RRESP_S = `AXI_RESP_OKAY;
          end
          default: begin
            RDATA_S = `AXI_DATA_BITS'd0;
            RRESP_S = `AXI_RESP_SLVERR; // error address
          end
        endcase

        if (RREADY_S) begin // master is ready to get data
          cs_slave_next = IDLE; // finish read request
          RID_S_next = `AXI_IDS_BITS'd0;
        end
      end

      WRITE_DATA: begin // write data
        WREADY_S = 1'b1; // always ready to write
        if (WVALID_S) begin
          // read data select
          if(addr_S_reg == `ASIC_ENABLE_OFFSET) begin
            ASIC_ENABLE_next = WDATA_S;
          end else if(addr_S_reg == `ASIC_MAPPING_PARAM_OFFSET) begin
            ASIC_MAPPING_PARAM_next = WDATA_S;
          end else if(addr_S_reg == `ASIC_SHAPE_PARAM1_OFFSET) begin
            ASIC_SHAPE_PARAM1_next = WDATA_S;
          end else if(addr_S_reg == `ASIC_SHAPE_PARAM2_OFFSET) begin
            ASIC_SHAPE_PARAM2_next = WDATA_S;
          end else if(addr_S_reg == `ASIC_IFMAP_ADDR_OFFSET) begin
            ASIC_IFMAP_ADDR_next = WDATA_S;
          end else if(addr_S_reg == `ASIC_FILTER_ADDR_OFFSET) begin
            ASIC_FILTER_ADDR_next = WDATA_S;
          end else if(addr_S_reg == `ASIC_BIAS_ADDR_OFFSET) begin
            ASIC_BIAS_ADDR_next = WDATA_S;
          end else if(addr_S_reg == `ASIC_OPSUM_ADDR_OFFSET) begin
            ASIC_OPSUM_ADDR_next = WDATA_S;
          end else if(addr_S_reg == `ASIC_GLB_FILTER_ADDR_OFFSET) begin
            ASIC_GLB_FILTER_ADDR_next = WDATA_S;
          end else if(addr_S_reg == `ASIC_GLB_OFMAP_ADDR_OFFSET) begin
            ASIC_GLB_OFMAP_ADDR_next = WDATA_S;
          end else if(addr_S_reg == `ASIC_GLB_BIAS_ADDR_OFFSET) begin
            ASIC_GLB_BIAS_ADDR_next = WDATA_S;
          end else if(addr_S_reg == `ASIC_IFMAP_LEN_OFFSET) begin
            ASIC_IFMAP_LEN_next = WDATA_S;
          end else if(addr_S_reg == `ASIC_OFMAP_LEN_OFFSET) begin
            ASIC_OFMAP_LEN_next = WDATA_S;
          end else begin
            write_error_next = 1'b1; // raise write error
          end

          if (WLAST_S) begin // if the write data is the last one
            cs_slave_next = WRITE_RESP; // write response
          end
        end
      end

      WRITE_RESP: begin
        BVALID_S = 1'b1;
        BRESP_S = (write_error)?`AXI_RESP_SLVERR:`AXI_RESP_OKAY;  // write response
        write_error_next = 1'b0; // reset

        if (BREADY_S) begin
          cs_slave_next = IDLE;
          BID_S_next = `AXI_IDS_BITS'd0;
        end
      end

      default: cs_slave_next = IDLE;
    endcase
  end

/*********************************************
        ASIC (Eyeriss Based PE-array)
*********************************************/

asic asic_0(
    .clk(ACLK),
    .rst(~ARESETn),

    .asic_interrupt(ASIC_interrupt),

    .asic_en(ASIC_ENABLE[0]),
    .maxpool_i(ASIC_ENABLE[1]),
    .relu_i(ASIC_ENABLE[2]),
    .operation_mode_i(ASIC_ENABLE[3]),
    .scaling_factor_i(ASIC_ENABLE[9:4]),

    /* mapping parameters */
    .m_i(ASIC_MAPPING_PARAM[25:16]), // number of ofmap channels stored in GLB
    .e_i(ASIC_MAPPING_PARAM[15:12]), // width of the PE sets
    .p_i(ASIC_MAPPING_PARAM[11:9]), // number of filters processed by a PE set
    .q_i(ASIC_MAPPING_PARAM[8:6]), // number of channels processed by a PE
    .r_i(ASIC_MAPPING_PARAM[5:3]), // number of PE sets that process different channels in the PE arrays
    .t_i(ASIC_MAPPING_PARAM[2:0]), // number of PE sets that process different filters in the PE arrays

    /* shape parameters */
    .C_i(ASIC_SHAPE_PARAM1[19:10]),
    .M_i(ASIC_SHAPE_PARAM1[9:0]),
    .W_i(ASIC_SHAPE_PARAM2[15:8]),
    .H_i(ASIC_SHAPE_PARAM2[7:0]),

    /* DRAM config */
    .ifmap_addr_i(ASIC_IFMAP_ADDR),
    .filter_addr_i(ASIC_FILTER_ADDR),
    .bias_addr_i(ASIC_BIAS_ADDR),
    .ofmap_addr_i(ASIC_OPSUM_ADDR),

    // staring address in GLB (Note: GLB_ifmap_addr = 0)
    .GLB_filter_addr_i(ASIC_GLB_FILTER_ADDR[`GLB_ADDR_BITS-1:0]),
    .GLB_bias_addr_i(ASIC_GLB_BIAS_ADDR[`GLB_ADDR_BITS-1:0]),
    .GLB_opsum_addr_i(ASIC_GLB_OFMAP_ADDR[`GLB_ADDR_BITS-1:0]),

    /* GLB */
    .GLB_EN(GLB_EN_asic),
    .GLB_WEB(GLB_WEB_asic),
    .GLB_MODE(GLB_MODE_asic),
    .GLB_A(GLB_A_asic),
    .GLB_DI(GLB_DI_asic),
    .GLB_DO(GLB_DO_asic),
    .GLB_mux(glb_mux),

    /* DMA */
    .DMA_en(DMA_EN),
    .DMA_mode(DMA_MODE),
    .DMA_byte_bias(DMA_BYTE_BIAS),
    .DMA_DRAM_ADDR(DMA_DRAM_ADDR),
    .DMA_GLB_ADDR(DMA_GLB_ADDR),
    .DMA_len(DMA_LEN),
    .DMA_done(DMA_DONE)
);


endmodule

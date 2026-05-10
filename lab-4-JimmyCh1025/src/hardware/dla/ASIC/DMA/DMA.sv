`include "AXI_define.svh"
`include "ASIC.svh"

module DMA (
    input clk,
    input rst,
    /* controller */
    input EN,
    input [1:0] MODE, // IFMAP:0, Filter:1, BIAS:2, OFMAP: 3
    input [1:0] BYTE_BIAS,
    output logic DONE,
    input [`AXI_ADDR_BITS-1:0] DRAM_ADDR,
    input [`GLB_ADDR_BITS-1:0] GLB_ADDR,
    input [`GLB_ADDR_BITS-1:0] LEN, // len = 0 means real

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
    output logic RREADY_M,

    /* GLB */
    output logic GLB_EN,  // active low
    output logic GLB_WEB, // active low
    output logic GLB_MODE, // Word:0, Byte:1
    output logic [`GLB_ADDR_BITS-1:0] GLB_A, // Byte Address (64KB, 16bits)
    output logic [`DATA_BITS-1:0] GLB_DI,
    input [`DATA_BITS-1:0] GLB_DO
);

/* AXI master FSM */
typedef enum logic [2:0] {
    IDLE,
    WRITE_ADDR,
    WRITE_DATA,
    WRITE_RESP,
    READ_ADDR,
    READ_DATA,
    DONE_S
} AXI_state;

/* GLB state */
typedef enum logic [2:0] {
    IDLE_GLB,
    WRITE_FILTER_GLB,
    WRITE_IFMAP_GLB,
    WRITE_BIAS_GLB,
    READ_OFMAP_ADDR_GLB,
    READ_OFMAP_GLB,
    DONE_S_GLB
} GLB_state;

/*************************** DMA config ***************************/
  logic DMAEN;
  logic [`GLB_ADDR_BITS-1:0] DMAGLB_ADDR;
  logic [`AXI_ADDR_BITS-1:0] DMADRAM_ADDR;
  logic [`GLB_ADDR_BITS-1:0] DMA_WORD_LEN;
  logic [1:0]DMABYTE_BIAS;
  logic [1:0]DMAMODE;
  logic DMADIR;

  logic [`AXI_ADDR_BITS-1:0] dram_addr;
  logic [`AXI_LEN_BITS-1:0] burst_len;
  logic [`AXI_SIZE_BITS-1:0] burst_size;
  logic AXI_enable;
  logic read_done;
  logic write_done;

  logic [`AXI_ADDR_BITS-1:0] glb_addr;
  logic GLB_enable;
  logic GLB_done;

  logic DIR;
  assign DIR = (MODE == `MODE_OFMAP)?1'b1:1'b0;

  always_ff @(posedge clk) begin
    if(rst) begin
      DMAEN <=`AXI_ADDR_BITS'd0;
      DMAGLB_ADDR <=`AXI_ADDR_BITS'd0;
      DMADRAM_ADDR <=`AXI_ADDR_BITS'd0;
      DMA_WORD_LEN <=`AXI_ADDR_BITS'd0;
      DMAMODE <= 2'd0;
      DMABYTE_BIAS <= 2'd0;
      DMADIR <= 1'b0;
    end else begin
      DMAEN <= (DONE)?1'b0:EN;
      DMAGLB_ADDR <= (EN)?GLB_ADDR:DMAGLB_ADDR;
      DMADRAM_ADDR <= (EN)?DRAM_ADDR:DMADRAM_ADDR;
      DMA_WORD_LEN <= (EN)?LEN:DMA_WORD_LEN;
      DMAMODE <= (EN)?MODE:DMAMODE;
      DMABYTE_BIAS <= (EN)?BYTE_BIAS:DMABYTE_BIAS;
      DMADIR <= (EN)?DIR:DMADIR;
    end
  end

  /*************************** DMA controller ***************************/
  DMA_controller DMA_controller_0(
    .clk(clk), // System clock
    .rst(rst), // System reset (active high)
    .DMAEN(DMAEN), // Enable the DMA
    .DMAGLB_ADDR(DMAGLB_ADDR), // Source address of DMA
    .DMADRAM_ADDR(DMADRAM_ADDR), // Destination address of DMA
    .DMALEN({{(`AXI_ADDR_BITS-`GLB_ADDR_BITS){1'b0}},DMA_WORD_LEN}), // Total length of the data
    .DMA_done(DONE),

    /* AXI control */
    .dram_addr(dram_addr),
    .burst_len(burst_len),
    .burst_size(burst_size),
    .AXI_enable(AXI_enable),
    .AXI_done(read_done|write_done),

    /* GLB control */
    .glb_addr(glb_addr[`GLB_ADDR_BITS-1:0]),
    .GLB_enable(GLB_enable),
    .GLB_done(GLB_done)
  );

    /*************************** DMA FIFO ***************************/

    logic FIFO_push_i_R, FIFO_pop_i_G;
    logic FIFO_push_i_G, FIFO_pop_i_W;
    logic [`AXI_DATA_BITS-1:0] FIFO_data_i_R, FIFO_data_i_G,  FIFO_data_o;
    logic FIFO_full, FIFO_empty;

    DMA_FIFO DMA_FIFO_0(
      .clk(clk),
      .rst(rst),
      .push_i((DMADIR)?FIFO_push_i_G:FIFO_push_i_R),
      .pop_i((DMADIR)?FIFO_pop_i_W:FIFO_pop_i_G),
      .data_i((DMADIR)?FIFO_data_i_G:FIFO_data_i_R),
      .data_o(FIFO_data_o),
      .full(FIFO_full),
      .empty(FIFO_empty)
    );

    /*************************** DMA master (Read/Write) ***************************/
    /* AXI master FSM */
    AXI_state cs_master_R, cs_master_R_next;
    AXI_state cs_master_W, cs_master_W_next;
    GLB_state cs_glb, cs_glb_next;

    assign read_done = (cs_master_R == DONE_S);
    assign write_done = (cs_master_W == DONE_S);
    assign GLB_done = (cs_glb == DONE_S_GLB);

    /*************************************
          read channel (DRAM -> GLB )
    *************************************/

    // Sequential logic
    always_ff @(posedge clk) begin
      if (rst) begin
        cs_master_R <= IDLE;
      end
      else begin
        cs_master_R <= cs_master_R_next;
      end
    end

    // Combinational logic
    always_comb begin
      // Default assignments
      cs_master_R_next = cs_master_R;

      // FIFO defaults signals
      FIFO_push_i_R = 1'b0;
      FIFO_data_i_R = RDATA_M; // read data -> FIFO

      // AXI interface defaults
      ARADDR_M = dram_addr; // read address
      ARID_M = `AXI_IDS_BITS'd0;
      ARLEN_M = burst_len; // burst length
      ARSIZE_M = burst_size; // burst size
      ARBURST_M = `AXI_BURST_INC; // increase mode
      ARVALID_M = 1'b0; // default low
      RREADY_M = 1'b0;

      case (cs_master_R)
        IDLE: begin
          if (AXI_enable && (DMADIR == 1'b0)) begin
            cs_master_R_next = READ_ADDR;
          end
        end
        READ_ADDR: begin
          ARVALID_M = 1'b1;
          if (ARREADY_M) begin
            cs_master_R_next = READ_DATA;
          end
        end
        READ_DATA: begin
          if(RVALID_M && !FIFO_full) begin
            FIFO_push_i_R = 1'b1; // push
            RREADY_M = 1'b1; // handshake
            if(RLAST_M) begin
              cs_master_R_next = DONE_S;
            end
          end
        end
        DONE_S: begin
          if(read_done) cs_master_R_next = IDLE; // wait for write channel
        end
        default: begin
          cs_master_R_next = IDLE;
        end
      endcase
    end

  /*************************************
                write channel
  *************************************/

  logic [`AXI_LEN_BITS-1:0] burst_count, burst_count_next;

  // Sequential logic
  always_ff @(posedge clk) begin
    if (rst) begin
      cs_master_W <= IDLE;
      burst_count <= `AXI_LEN_BITS'd0;
    end
    else begin
      cs_master_W <= cs_master_W_next;
      burst_count <= burst_count_next;
    end
  end

  // Combinational logic
  always_comb begin
    // Default assignments
    cs_master_W_next = cs_master_W;
    burst_count_next = burst_count;

    // FIFO defaults signals
    FIFO_pop_i_W = 1'b0;

    // AXI interface defaults
    AWADDR_M = dram_addr; // write address
    AWID_M = `AXI_IDS_BITS'd0;
    AWLEN_M = burst_len; // burst length
    AWSIZE_M = burst_size; // burst size
    AWBURST_M = `AXI_BURST_INC; // increase mode
    AWVALID_M = 1'b0; // default low

    WDATA_M = FIFO_data_o; // FIFO -> write data
    WLAST_M = 1'b0;
    WSTRB_M = `AXI_STRB_WORD; // always words
    WVALID_M = 1'b0;

    BREADY_M = 1'b0;

    case (cs_master_W)
      IDLE: begin
        // NOTE: ensure that the read channel active first
        if (AXI_enable && (DMADIR == 1'b1)) begin
          cs_master_W_next = WRITE_ADDR;
        end
      end
      WRITE_ADDR: begin
        AWVALID_M = 1'b1;
        if(AWREADY_M) begin
          cs_master_W_next = WRITE_DATA;
          burst_count_next = `AXI_LEN_BITS'd0;
        end
      end
      WRITE_DATA: begin
        WLAST_M = (burst_count == burst_len)?1'b1:1'b0;
        WVALID_M = !FIFO_empty; // active if FIFO is not empty
        if (WREADY_M & !FIFO_empty) begin
          FIFO_pop_i_W = 1'b1; // pop
          if(burst_count == burst_len) begin // the last one
            cs_master_W_next = WRITE_RESP;
            burst_count_next = `AXI_LEN_BITS'd0;
          end else begin
            burst_count_next = burst_count + `AXI_LEN_BITS'd1;
          end
        end
      end
      WRITE_RESP: begin
        BREADY_M = 1'b1;
        if(BVALID_M && (BRESP_M == `AXI_RESP_OKAY)) begin
          cs_master_W_next = DONE_S;
        end
      end
      DONE_S: begin
        if(write_done) cs_master_W_next = IDLE; // wait for write channel
      end
      default: begin
        cs_master_W_next = IDLE;
      end
    endcase
  end

  /*************************** GLB I/O state machine ***************************/

    // IDLE,
    // WRITE_FILTER,
    // WRITE_IFMAP,
    // WRITE_BIAS,
    // READ_OFMAP_ADDR,
    // READ_OFMAP,
    // DONE_S

    // output logic GLB_EN,  // active low
    // output logic GLB_WEB, // active low
    // output logic GLB_MODE, // Word:0, Byte:1
    // output logic [`GLB_ADDR_BITS-1:0] GLB_A, // Byte Address (64KB, 16bits)
    // output logic [`DATA_BITS-1:0] GLB_DI,
    // input [`DATA_BITS-1:0] GLB_DO

  logic [`GLB_ADDR_BITS-3:0] GLB_A_word;
  logic [`GLB_ADDR_BITS-3:0] GLB_A_word_next;
  logic [1:0] GLB_A_bias;
  logic [1:0] GLB_A_bias_next;

  logic [`GLB_ADDR_BITS-3:0] counter_R, counter_W;
  logic [`GLB_ADDR_BITS-3:0] counter_R_next, counter_W_next;

  assign GLB_A = {GLB_A_word, GLB_A_bias};

  always_ff @(posedge clk) begin
    if(rst) begin
      cs_glb <= IDLE_GLB;
      GLB_A_word <= {(`GLB_ADDR_BITS-3){1'd0}};
      GLB_A_bias <= 2'd0;
      counter_R <=  `GLB_ADDR_BITS'd0;
      counter_W <=  `GLB_ADDR_BITS'd0;
    end else begin
      cs_glb <= cs_glb_next;
      GLB_A_word <= GLB_A_word_next;
      GLB_A_bias <= GLB_A_bias_next;
      counter_R <= counter_R_next;
      counter_W <= counter_W_next;
    end
  end

  // FIFO data out -> GLB data in
  always_comb begin
    if(GLB_MODE == `BYTE_MODE) begin
      case (counter_R[1:0]) // select FIFO output data
        2'd0:GLB_DI = {24'd0,FIFO_data_o[7:0]};
        2'd1:GLB_DI = {24'd0,FIFO_data_o[15:8]};
        2'd2:GLB_DI = {24'd0,FIFO_data_o[23:16]};
        2'd3:GLB_DI = {24'd0,FIFO_data_o[31:24]};
      endcase
    end else begin
      GLB_DI = FIFO_data_o;
    end
  end

  always_comb begin
    cs_glb_next = cs_glb;
    GLB_EN = 1'b1;
    GLB_WEB = 1'b1;
    GLB_MODE = `WORD_MODE;
    GLB_A_word_next = GLB_A_word;
    GLB_A_bias_next = GLB_A_bias;
    counter_R_next = counter_R;
    counter_W_next = counter_W;


    FIFO_pop_i_G = 1'b0;
    FIFO_push_i_G = 1'b0;
    FIFO_data_i_G = GLB_DO;

    case (cs_glb)
      IDLE_GLB:begin
        if(GLB_enable)begin
          GLB_A_word_next = glb_addr[`GLB_ADDR_BITS-1:2];
          counter_R_next = `GLB_ADDR_BITS'd0;
          counter_W_next = `GLB_ADDR_BITS'd0;
          case (DMAMODE)
            `MODE_IFMAP:begin // read, byte, mod by len
              cs_glb_next = WRITE_IFMAP_GLB;
              GLB_A_bias_next = DMABYTE_BIAS;
            end
            `MODE_FILTER:begin // read, byte, bias
              cs_glb_next = WRITE_FILTER_GLB;
              GLB_A_bias_next = 2'd0;
            end
            `MODE_BIAS:begin // read, word
              cs_glb_next = WRITE_BIAS_GLB;
              GLB_A_bias_next = 2'd0;
            end
            `MODE_OFMAP:begin // write, word
              cs_glb_next = READ_OFMAP_ADDR_GLB;
              GLB_A_bias_next = 2'd0;
            end
          endcase
        end
      end
      WRITE_FILTER_GLB:begin
        GLB_EN = 1'b0; // enable
        GLB_MODE = `BYTE_MODE;

        if(!FIFO_empty)begin
          GLB_WEB = 1'b0; // write
          // read from FIFO
          if(counter_R == `GLB_ADDR_BITS'd3)begin
            FIFO_pop_i_G = 1'b1; // pop out to switch the data
            counter_R_next = `GLB_ADDR_BITS'd0;
          end else begin
            counter_R_next = counter_R + `GLB_ADDR_BITS'd1;
          end

          // write to global buffer
          if(counter_W + `GLB_ADDR_BITS'd1 == DMA_WORD_LEN) begin
            if(GLB_A_bias == 2'd3) begin
              cs_glb_next = DONE_S_GLB;
              GLB_A_bias_next = 2'd0;
            end else begin
              GLB_A_bias_next = (GLB_A_bias+2'd1);
              counter_W_next = `GLB_ADDR_BITS'd0;
              GLB_A_word_next = glb_addr[`GLB_ADDR_BITS-1:2];
            end
          end else begin
            counter_W_next = counter_W + `GLB_ADDR_BITS'd1;
            GLB_A_word_next = GLB_A_word + `GLB_ADDR_BITS'd1;
          end
        end
      end
      WRITE_IFMAP_GLB:begin
        GLB_EN = 1'b0; // enable
        GLB_MODE = `BYTE_MODE;

        if(!FIFO_empty)begin
          GLB_WEB = 1'b0; // write
          // read from FIFO
          if(counter_R == `GLB_ADDR_BITS'd3)begin
            FIFO_pop_i_G = 1'b1; // pop out to switch the data
            counter_R_next = `GLB_ADDR_BITS'd0;
          end else begin
            counter_R_next = counter_R + `GLB_ADDR_BITS'd1;
          end

          // write to global buffer
          if(counter_W + `GLB_ADDR_BITS'd1 == {DMA_WORD_LEN[`GLB_ADDR_BITS-3:0],2'b00}) begin
            cs_glb_next = DONE_S_GLB;
          end else begin
            counter_W_next = counter_W + `GLB_ADDR_BITS'd1;
            GLB_A_word_next = GLB_A_word + `GLB_ADDR_BITS'd1;
          end
        end
      end
      WRITE_BIAS_GLB:begin
        GLB_EN = 1'b0; // enable
        GLB_MODE = `WORD_MODE;
        if(!FIFO_empty)begin
          GLB_WEB = 1'b0; // write

          FIFO_pop_i_G = 1'b1; // pop out to switch the data

          // write to global buffer
          if(counter_W + `GLB_ADDR_BITS'd1 == DMA_WORD_LEN) begin
            cs_glb_next = DONE_S_GLB;
          end else begin
            counter_W_next = counter_W + `GLB_ADDR_BITS'd1;
            GLB_A_word_next = GLB_A_word + `GLB_ADDR_BITS'd1;
          end
        end
      end
      READ_OFMAP_ADDR_GLB:begin
        GLB_EN = 1'b0; // enable
        GLB_MODE = `WORD_MODE;
        cs_glb_next = READ_OFMAP_GLB;
      end
      READ_OFMAP_GLB:begin
        GLB_EN = 1'b0; // enable
        GLB_MODE = `WORD_MODE;
        if(!FIFO_full) begin
          FIFO_push_i_G = 1'b1;
          if(counter_R + `GLB_ADDR_BITS'd1 == DMA_WORD_LEN)begin
            cs_glb_next = DONE_S_GLB;
          end else begin
            cs_glb_next = READ_OFMAP_ADDR_GLB;
            counter_R_next = counter_R + `GLB_ADDR_BITS'd1;
            GLB_A_word_next = GLB_A_word + `GLB_ADDR_BITS'd1;
          end
        end
      end
      DONE_S_GLB:begin
        if(GLB_enable == 1'b0) cs_glb_next = IDLE_GLB; // wait for AXI W/R channel
      end
      default:begin
        cs_glb_next = IDLE_GLB;
      end
    endcase
  end

endmodule

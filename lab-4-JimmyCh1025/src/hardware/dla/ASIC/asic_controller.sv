`include "AXI_define.svh"
`include "ASIC.svh"

module asic_controller (
    input clk,
    input rst,
    input asic_en,
    output logic asic_done,
    /* MMIO */
    input [`AXI_ADDR_BITS-1:0] ifmap_addr,
    input [`AXI_ADDR_BITS-1:0] filter_addr,
    input [`AXI_ADDR_BITS-1:0] bias_addr,
    input [`AXI_ADDR_BITS-1:0] ofmap_addr,
    input [`GLB_ADDR_BITS-1:0] GLB_filter_addr,
    input [`GLB_ADDR_BITS-1:0] GLB_bias_addr,
    input [`GLB_ADDR_BITS-1:0] GLB_opsum_addr,
    /* Layer Info */
    input maxpool,
    /* mapping parameters */
    input [9:0] m,
    input [3:0] e,
    input [2:0] p,
    input [2:0] q,
    input [2:0] r,
    input [2:0] t,
    /* shape parameters */
    input [9:0] C,
    input [9:0] M,
    input [7:0] W,
    input [7:0] H,
    /* DMA */
    output logic DMA_en,
    output logic [1:0] DMA_mode,
    output logic [`AXI_ADDR_BITS-1:0] DMA_DRAM_ADDR,
    output logic [`GLB_ADDR_BITS-1:0] DMA_GLB_ADDR,
    output logic [`GLB_ADDR_BITS-1:0] DMA_len,
    output logic [1:0] DMA_byte_bias,
    input DMA_done,
    /* ID config */
    output logic set_XID,
    output logic [`XID_BITS-1:0] ifmap_XID_scan_in,
    output logic [`XID_BITS-1:0] filter_XID_scan_in,
    output logic [`XID_BITS-1:0] ipsum_XID_scan_in,
    output logic [`XID_BITS-1:0] opsum_XID_scan_in,
    output logic set_YID,
    output logic [`YID_BITS-1:0] ifmap_YID_scan_in,
    output logic [`YID_BITS-1:0] filter_YID_scan_in,
    output logic [`YID_BITS-1:0] ipsum_YID_scan_in,
    output logic [`YID_BITS-1:0] opsum_YID_scan_in,
    output logic set_LN,
    output logic [`PE_ARRAY_H-2:0] LN_config_in,

    /* PE_Array */
    output logic [`PE_ARRAY_H*`PE_ARRAY_W-1:0] PE_en,
    output logic [10:0] PE_config,

    output logic PEA_ifmap_valid,
    input PEA_ifmap_ready,
    output logic [`XID_BITS-1:0] ifmap_tag_X,
    output logic [`YID_BITS-1:0] ifmap_tag_Y,


    output logic PEA_filter_valid,
    input PEA_filter_ready,
    output logic [`XID_BITS-1:0] filter_tag_X,
    output logic [`YID_BITS-1:0] filter_tag_Y,

    output logic PEA_ipsum_valid,
    input PEA_ipsum_ready,
    output logic [`XID_BITS-1:0] ipsum_tag_X,
    output logic [`YID_BITS-1:0] ipsum_tag_Y,

    input PEA_opsum_valid,
    output logic PEA_opsum_ready,
    output logic [`XID_BITS-1:0] opsum_tag_X,
    output logic [`YID_BITS-1:0] opsum_tag_Y,

    /* GLB */
    output logic GLB_EN,
    output logic GLB_WEB,
    output logic GLB_MODE,
    output logic [`GLB_ADDR_BITS-1:0] GLB_A,
    output logic GLB_mux,
    output logic GLB_DI_select,
    output logic GLB_DO_select,

    /* PPU */
    output logic relu_sel,
    output logic Maxpool_en,
    output logic Maxpool_init

);

/* declare signals */
    typedef enum logic [4:0] {
        IDLE, //0
        SEND_CONFIG_SCAN, // 1
        LOAD_BIAS, //2
        LOAD_IFMAP, //3
        LOAD_FILT, //4
        SEND_PE_CONFIG, //5
        SEND_AFILT, //6
        SEND_FILT, //7
        SEND_AIFMAP, //8
        SEND_IFMAP, //9
        SEND_AIPSUM, //10
        SEND_IPSUM, //11
        STORE_OPSUM, //12
        UPDATE, //13
        PASS_PPU, //14
        STORE_OFMAP, //15
        SEND_OFMAP, //16
        DONE // 17
    } state_t;
    state_t cs, ns;

    parameter MEM_TO_GLB = 1'b0,
              GLB_TO_MEM = 1'b1;

    logic [2:0] count_x;
    logic [1:0] count_y1;
    logic count_y2;

    logic [1:0] count_c;
    logic [3:0] count_filt;
    logic [2:0] count_PP_filt;
    logic [7:0] count_W;
    logic [7:0] count_F;
    logic [7:0] PPU_count_F;
    logic [3:0] count_h;
    logic [7:0] count_H;
    logic [9:0] count_m;
    logic [9:0] count_C;
    logic [7:0] count_E;
    logic [9:0] count_M;
    logic [2:0] count_e;
    logic [2:0] count_PPU;

    logic count_PP_filt_reset;
    logic count_h_reset;
    logic count_e_reset;
    logic count_F_reset;
    logic count_C_reset;
    logic count_PPU_reset;
    logic count_m_reset;
    logic count_E_reset;
    logic count_M_reset;

    logic H_top_padding;
    logic H_down_pdding;
    logic count_H_reset;
    logic count_c_reset;



    logic [3:0] valid_e;

    logic [`AXI_ADDR_BITS-1:0] mul1_src1;
    logic [`AXI_ADDR_BITS-1:0] mul1_src2;
    logic [`AXI_ADDR_BITS-1:0] mul1_res;

    logic [`AXI_ADDR_BITS-1:0] mul2_src1;
    logic [`AXI_ADDR_BITS-1:0] mul2_src2;
    logic [`AXI_ADDR_BITS-1:0] mul2_res;

    logic [`AXI_ADDR_BITS-1:0] mul3_src1;
    logic [`AXI_ADDR_BITS-1:0] mul3_src2;
    logic [`AXI_ADDR_BITS-1:0] mul3_res;

    logic [2:0] count_str_m;
    logic [2:0] count_str_f;
    logic [2:0] count_str_e;

    logic [7:0] temp_valid_e;
    logic [3:0] h_max;

    logic [5:0] PP_filt;
    logic [5:0] PP_ch;

    logic [7:0] E;
    logic [7:0] F;
    logic [11:0] eF;
    logic [15:0] EF;
    logic [13:0] C9;
    logic no_ipsum;

    logic [7:0] PE_F;
    logic [2:0] PE_config_input_ch;

/* FSM */
/* ================================================================================================= */
    always_ff @( posedge clk ) begin
        if (rst) cs <= IDLE;
        else cs <= ns;
    end

    always_comb  begin
        ns = cs;
        case (cs)
            IDLE: begin
                if (asic_en) ns = SEND_CONFIG_SCAN;
            end
            SEND_CONFIG_SCAN: begin
                if (count_x == ('d`PE_ARRAY_W-1) && count_y1 == ('d`FILT_S-1) && count_y2) ns = LOAD_BIAS;
            end
            LOAD_BIAS: begin
                if (DMA_done) ns = LOAD_IFMAP;
            end
            LOAD_IFMAP: begin
                if (DMA_done && count_c_reset) ns = LOAD_FILT;
            end
            LOAD_FILT: begin
                if (DMA_done && count_PP_filt_reset) ns = SEND_PE_CONFIG;
            end
            SEND_PE_CONFIG: ns = SEND_AFILT;
            SEND_AFILT: ns = SEND_FILT;
            SEND_FILT: begin
                if (PEA_filter_ready && count_filt == 4'd8 && count_PP_filt_reset) ns = SEND_AIFMAP;
                else if (PEA_filter_ready) ns = SEND_AFILT;
            end
            SEND_AIFMAP: ns = SEND_IFMAP;
            SEND_IFMAP: begin
                if (PEA_ifmap_ready && count_h_reset && count_W > 8'd1) ns = SEND_AIPSUM;
                else if (PEA_ifmap_ready) ns = SEND_AIFMAP;
            end
            SEND_AIPSUM: ns = SEND_IPSUM;
            SEND_IPSUM: begin
                if (PEA_ipsum_ready && count_PP_filt_reset && count_e_reset) ns = STORE_OPSUM;
                else if (PEA_ipsum_ready) ns = SEND_AIPSUM;
            end
            STORE_OPSUM: begin
                if (PEA_opsum_valid && count_PP_filt_reset && count_e_reset) begin
                    if (count_F_reset) begin
                        if (count_C_reset) ns = PASS_PPU;
                        else ns = UPDATE;
                    end
                    else ns = SEND_AIFMAP;
                end
            end
            PASS_PPU: begin
                if (count_PPU_reset) ns = STORE_OFMAP;
            end
            STORE_OFMAP: begin
                if (count_F_reset && count_e_reset && count_PP_filt_reset) ns = SEND_OFMAP;
                else ns = PASS_PPU;
            end
            SEND_OFMAP: begin
                if (DMA_done && count_PP_filt_reset) ns = UPDATE;
            end
            UPDATE: begin
                if (count_m_reset) begin
                    if (count_C_reset && count_E_reset) begin
                        if (count_M_reset) ns = DONE;
                        else ns = LOAD_BIAS;
                    end
                    else ns = LOAD_IFMAP;
                end
                else ns = LOAD_FILT;
            end
            DONE: if (!asic_en) ns = IDLE;
        endcase
    end

/* DONE */
/* ================================================================================================= */
assign asic_done = (cs == DONE);

/* DMA */
/* ================================================================================================= */
    assign DMA_en = (cs == LOAD_IFMAP || cs == LOAD_FILT || cs == LOAD_BIAS || cs == SEND_OFMAP);
    assign DMA_mode = (cs == LOAD_IFMAP)? 2'd`MODE_IFMAP: (cs == LOAD_FILT)? 2'd`MODE_FILTER: (cs == LOAD_BIAS)? 2'd`MODE_BIAS: (cs == SEND_OFMAP)? 2'd`MODE_OFMAP: 2'd0;
    always_comb  begin // DMA_DRAM_ADDR
        case(cs)
            LOAD_IFMAP: DMA_DRAM_ADDR = ifmap_addr + mul1_res + mul2_res;
            LOAD_FILT: DMA_DRAM_ADDR = filter_addr + mul1_res + mul2_res;
            LOAD_BIAS: DMA_DRAM_ADDR = bias_addr + {count_M, 2'd0};
            SEND_OFMAP: DMA_DRAM_ADDR = ofmap_addr + mul1_res + mul2_res;
            default: DMA_DRAM_ADDR = 'd0;
        endcase
    end
    always_comb  begin // DMA_GLB_ADDR
        case(cs)
            LOAD_IFMAP: DMA_GLB_ADDR = 'd0;
            LOAD_FILT: DMA_GLB_ADDR = GLB_filter_addr + {mul3_res, 2'd0};
            LOAD_BIAS: DMA_GLB_ADDR = GLB_bias_addr;
            SEND_OFMAP: DMA_GLB_ADDR = GLB_opsum_addr + mul3_res;
            default: DMA_GLB_ADDR = 'd0;
        endcase
    end
    always_comb  begin // DMA_len
        case(cs)
            LOAD_IFMAP: DMA_len = {2'd0,mul3_res[31:2]};
            LOAD_FILT: DMA_len = 'd9;
            LOAD_BIAS: DMA_len = m;
            SEND_OFMAP: DMA_len = (maxpool)? {4'd0,eF[11:4]}:{2'd0,eF[11:2]};
            default: DMA_len = 'd0;
        endcase
    end
    assign DMA_byte_bias = count_c;

/* GLB */
/* ================================================================================================= */

    assign GLB_mux = (cs == LOAD_IFMAP || cs == LOAD_FILT || cs == LOAD_BIAS || cs == SEND_OFMAP)? `DMA: `ASIC;
    assign GLB_EN = ~(cs == SEND_AIFMAP || cs == SEND_IFMAP || cs == SEND_AFILT || cs == SEND_FILT
                       || cs == SEND_AIPSUM || cs == SEND_IPSUM || cs == STORE_OPSUM || cs == PASS_PPU || cs == STORE_OFMAP);
    assign GLB_WEB = ~(cs == STORE_OPSUM || cs == STORE_OFMAP);
    always_comb  begin // GLB_A
        if (maxpool) begin
            case(count_PPU)
                3'd0: PPU_count_F = {count_W,1'b0};
                3'd1: PPU_count_F = {count_W,1'b1};
                3'd2: PPU_count_F = {count_W,1'b0};
                3'd3: PPU_count_F = {count_W,1'b1};
                default: PPU_count_F = 'd0;
            endcase
        end
        else PPU_count_F = count_W;

        case(cs)
            SEND_AIFMAP, SEND_IFMAP: GLB_A = {(count_W - 14'd1 + mul1_res), 2'd0};
            SEND_AFILT, SEND_FILT: GLB_A = GLB_filter_addr + {(count_filt + mul1_res), 2'd0};
            SEND_AIPSUM, SEND_IPSUM: begin
                if (no_ipsum) GLB_A = GLB_bias_addr + {(count_m + count_PP_filt), 2'd0};
                else GLB_A = GLB_opsum_addr + {(count_F + mul1_res + mul2_res), 2'd0};
            end
            STORE_OPSUM: GLB_A = GLB_opsum_addr + {(count_F + mul1_res + mul2_res), 2'd0};
            STORE_OFMAP: GLB_A = GLB_opsum_addr + (count_W + mul1_res + mul2_res);
            PASS_PPU: GLB_A = GLB_opsum_addr + {(PPU_count_F + mul1_res + mul2_res), 2'd0};
            default: GLB_A = 'd0;
        endcase
    end
    assign GLB_MODE = (cs == LOAD_IFMAP || cs == LOAD_FILT || cs == STORE_OFMAP)? `BYTE_MODE: `WORD_MODE;
    assign GLB_DI_select = (cs == STORE_OPSUM)? 1'b`GLB_DO_PSUM: 1'b`GLB_DO_OFMAP;
    assign GLB_DO_select = (cs == SEND_IFMAP && (count_W == 8'b0 || count_W == (W - 8'd1) || (count_E == 8'b0 && count_h == 4'b0) || (count_E_reset && count_h_reset)))? `WITH_PAD: `NO_PAD;


/* count */
/* ================================================================================================= */

    assign count_c_reset = (count_c == PP_ch - 2'd1);
    always_ff @( posedge clk ) begin // count_c
        if (rst) count_c <= 2'b00;
        else begin
            if (cs == LOAD_IFMAP && DMA_done) begin
                if (count_c_reset) count_c <= 2'd0;
                else count_c <= count_c + 2'd1;
            end
        end
    end

    always_ff @( posedge clk ) begin // count_filt
        if (rst) count_filt <= 4'd0;
        else begin
            if (cs == SEND_FILT && PEA_filter_ready) begin
                if (count_filt == 4'd8) count_filt <= 4'd0;
                else count_filt <= count_filt + 4'd1;
            end
        end
    end

    assign count_PP_filt_reset = (count_PP_filt == PP_filt - 4'd1);

    always_ff @( posedge clk ) begin // count_PP_filt
        if (rst) count_PP_filt <= 'd0;
        else begin
            if (cs == LOAD_FILT && DMA_done) begin
                if (count_PP_filt_reset) count_PP_filt <= 'd0;
                else count_PP_filt <= count_PP_filt + 'd1;
            end
            else if (cs == SEND_FILT && PEA_filter_ready) begin
                if (count_filt == 4'd8) begin
                    if (count_PP_filt_reset) count_PP_filt <= 'd0;
                    else count_PP_filt <= count_PP_filt + 'd1;
                end
            end
            else if (cs == SEND_IPSUM && PEA_ipsum_ready) begin
                if (count_PP_filt_reset) count_PP_filt <= 'd0;
                else count_PP_filt <= count_PP_filt + 'd1;
            end
            else if (cs == STORE_OPSUM && PEA_opsum_valid) begin
                if (count_PP_filt_reset) count_PP_filt <= 'd0;
                else count_PP_filt <= count_PP_filt + 'd1;
            end
            else if (cs == STORE_OFMAP && count_e_reset && count_F_reset) begin
                if (count_PP_filt_reset) count_PP_filt <= 'd0;
                else count_PP_filt <= count_PP_filt + 'd1;
            end
            else if (cs == SEND_OFMAP && DMA_done) begin
                if (count_PP_filt_reset) count_PP_filt <= 'd0;
                else count_PP_filt <= count_PP_filt + 'd1;
            end
        end
    end

    assign count_F = count_W - 8'd2;
    assign count_F_reset = (cs == STORE_OFMAP)?((maxpool)? (count_W == F[7:1] - 8'd1): (count_W == F - 1)): (count_W == W - 1);
    always_ff @( posedge clk ) begin // count_W
        if (rst) count_W <= 'd0;
        else begin
            if (cs == SEND_IFMAP && PEA_ifmap_ready && count_h_reset) begin
                if (count_W < 8'd2) count_W <= count_W + 8'd1;
            end
            if (cs == STORE_OPSUM && PEA_opsum_valid && count_PP_filt_reset && count_e_reset) begin
                if (count_F_reset) count_W <= 'd0;
                else count_W <= count_W + 'd1;
            end
            else if (cs == STORE_OFMAP) begin
                if (count_F_reset) count_W <= 'd0;
                else count_W <= count_W + 'd1;
            end
        end
    end

    assign count_h_reset = (count_h == h_max - 1);
    always_ff @( posedge clk ) begin // count_h
        if (rst) count_h <= 4'd0;
        else begin
            if (cs == SEND_IFMAP && PEA_ifmap_ready) begin
                if (count_h_reset) count_h <= 4'd0;
                else count_h <= count_h + 4'd1;
            end
        end
    end

    assign count_m_reset = (count_m + PP_filt == m);
    always_ff @( posedge clk ) begin // count_m
        if (rst) count_m <= 10'd0;
        else begin
            if (cs == UPDATE) begin
                if (count_m_reset) count_m <= 10'd0;
                else count_m <= count_m + PP_filt;
            end
        end
    end

    assign count_C_reset = (count_C + PP_ch == C);
    always_ff @( posedge clk ) begin // count_C
        if (rst) count_C <= 10'd0;
        else begin
            if (cs == UPDATE && count_m_reset) begin
                if (count_C_reset) count_C <= 10'd0;
                else count_C <= count_C + PP_ch;
            end
        end
    end

    assign count_H_reset = (count_H + h_max + 8'd1 >= H);
    always_ff @( posedge clk ) begin // count_H
        if (rst) count_H <= 8'd0;
        else begin
            if (cs == UPDATE && count_m_reset && count_C_reset) begin
                if (count_H_reset) count_H <= 8'd0;
                else if (count_H == 3'd0) count_H <= count_H + valid_e - 8'd1;
                else count_H <= count_H + valid_e;
            end
        end
    end

    assign count_E_reset = (count_E + valid_e == E);
    always_ff @( posedge clk ) begin // count_E
        if (rst) count_E <= 8'd0;
        else begin
            if (cs == UPDATE && count_m_reset && count_C_reset) begin
                if (count_E_reset) count_E <= 8'd0;
                else count_E <= count_E + valid_e;
            end
        end
    end

    assign count_M_reset = (count_M + m == M);
    always_ff @( posedge clk ) begin // count_M
        if (rst) count_M <= 10'd0;
        else begin
            if (cs == UPDATE && count_m_reset && count_C_reset && count_E_reset) begin
                if (count_M_reset) count_M <= 10'd0;
                else count_M <= count_M + m;
            end
        end
    end

    assign count_e_reset = (cs == STORE_OFMAP && maxpool)? (count_e == valid_e[3:1] - 'd1): (count_e == valid_e - 'd1);
    always_ff @( posedge clk ) begin // count_e
        if (rst) count_e <= 3'd0;
        else begin
            if (cs == SEND_IPSUM && PEA_ipsum_ready && count_PP_filt_reset) begin
                if (count_e_reset) count_e <= 3'd0;
                else count_e <= count_e + 3'd1;
            end
            else if (cs == STORE_OPSUM && PEA_opsum_valid && count_PP_filt_reset) begin
                if (count_e_reset) count_e <= 3'd0;
                else count_e <= count_e + 3'd1;
            end
            else if (cs == STORE_OFMAP && count_F_reset) begin
                if (count_e_reset) count_e <= 3'd0;
                else count_e <= count_e + 3'd1;
            end
        end
    end

    always_ff @( posedge clk ) begin // count_x count_y1 count_y2
        if (rst) begin
            count_x <= 3'd0;
            count_y1 <= 2'd0;
            count_y2 <= 1'd0;
        end
        else begin
            if (cs == SEND_CONFIG_SCAN) begin
                if (count_x == 3'd7) begin
                    count_x <= 3'd0;
                    if (count_y1 == 2'd2) begin
                        count_y1 <= 2'd0;
                        count_y2 <= ~count_y2;
                    end
                    else count_y1 <= count_y1 + 2'd1;
                end
                else count_x <= count_x + 3'd1;
            end
        end
    end

    assign count_PPU_reset = (maxpool)? count_PPU == 3'd4: count_PPU == 3'd1;
    always_ff @( posedge clk ) begin // count_PPU
        if (rst) count_PPU <= 3'b0;
        else begin
            if (cs == PASS_PPU) begin
                if (count_PPU_reset) count_PPU <= 3'b0;
                else count_PPU <= count_PPU + 3'b1;
            end
        end
    end

/* tag */
/* ================================================================================================= */

    assign ifmap_tag_X = (cs == SEND_IFMAP)? count_h: `DEFAULT_XID;
    assign ifmap_tag_Y = (cs == SEND_IFMAP)? 2'd0: `DEFAULT_YID;

    always_comb  begin // filter_tag_X
        if (cs == SEND_FILT) begin
            if(count_filt < 4'd3) filter_tag_X = 4'd0;
            else if (count_filt < 4'd6) filter_tag_X = 4'd1;
            else filter_tag_X = 4'd2;
        end
        else filter_tag_X = `DEFAULT_XID;
    end
    always_comb  begin // filter_tag_Y
        if (cs == SEND_FILT) begin
            if (count_PP_filt > 3'd3) filter_tag_Y = 3'd1;
            else filter_tag_Y = 3'd0;
        end
        else filter_tag_Y = `DEFAULT_YID;
    end

    assign ipsum_tag_X = (cs == SEND_IPSUM)? count_e: `DEFAULT_XID;
    assign ipsum_tag_Y = (cs == SEND_IPSUM)? (count_PP_filt > 3'd3)? 3'd1: 3'd0: `DEFAULT_YID;


    assign opsum_tag_X = (cs == STORE_OPSUM)? count_e: `DEFAULT_XID;
    assign opsum_tag_Y = (cs == STORE_OPSUM)? (count_PP_filt > 3'd3)? 3'd1: 3'd0: `DEFAULT_YID;

/* input config process */
/* ================================================================================================= */
    assign PP_filt = p * t;
    assign PP_ch = (C < 'd4)? C: q * r;
    assign E = W - 2;
    assign F = H - 2;

/* Config Scan Chain Setup */
/* ================================================================================================= */
    assign set_XID = (cs == SEND_CONFIG_SCAN);
    assign ifmap_XID_scan_in = (count_x < e)? count_x + count_y1: `DEFAULT_XID;
    assign filter_XID_scan_in = (count_x < e)? count_y1: `DEFAULT_XID;
    assign ipsum_XID_scan_in = (count_x < e && count_y1 == 2'd0)? count_x: `DEFAULT_XID;
    assign opsum_XID_scan_in = (count_x < e && count_y1 == 2'd2)? count_x: `DEFAULT_XID;
    assign set_YID = (cs == SEND_CONFIG_SCAN && count_x == 3'd0);
    assign ifmap_YID_scan_in = (r == 2)? count_y2: 1'd0;
    assign filter_YID_scan_in = count_y2;
    assign ipsum_YID_scan_in = (count_y1 == 2'd0)? count_y2: `DEFAULT_YID;
    assign opsum_YID_scan_in = (count_y1 == 2'd2)? count_y2: `DEFAULT_YID;
    assign set_LN = (cs == SEND_CONFIG_SCAN && count_x == 'd0 && count_y1 == 'd0 && count_y2 == 'd0);
    assign LN_config_in = (r == 2)? 5'b11111: 5'b11011;

/* Misalignment Ofmap Row Process*/
/* ================================================================================================= */
    assign temp_valid_e = (count_E + 8'd`PE_ARRAY_W > E)? E - count_E: 8'd`PE_ARRAY_W;
    assign valid_e = temp_valid_e[3:0];
    assign h_max = valid_e+4'd2;
    assign no_ipsum = (count_C == 10'd0);

/* Send PE Config */
/* ================================================================================================= */
    always_comb  begin
        if (cs == SEND_PE_CONFIG) begin
            case (valid_e)
                4'd1: PE_en = {6{8'b10000000}};
                4'd2: PE_en = {6{8'b11000000}};
                4'd3: PE_en = {6{8'b11100000}};
                4'd4: PE_en = {6{8'b11110000}};
                4'd5: PE_en = {6{8'b11111000}};
                4'd6: PE_en = {6{8'b11111100}};
                4'd7: PE_en = {6{8'b11111110}};
                4'd8: PE_en = {6{8'b11111111}};
                default: PE_en = 'd0;
            endcase
        end
        else PE_en = 'd0;
    end
    assign PE_F = F - 'd1;
    assign PE_config_input_ch = PP_ch - 3'd1;
    assign PE_config = (cs == SEND_PE_CONFIG)? {1'b`CONV, 2'd3, PE_F[4:0], PE_config_input_ch[1:0]} :10'd0;


/* valid ready signals */
/* ================================================================================================= */
    assign PEA_ifmap_valid = (cs == SEND_IFMAP)? 1'b1: 1'b0;
    assign PEA_filter_valid = (cs == SEND_FILT)? 1'b1: 1'b0;
    assign PEA_ipsum_valid = (cs == SEND_IPSUM)? 1'b1: 1'b0;
    assign PEA_opsum_ready = (cs == STORE_OPSUM)? 1'b1: 1'b0;

/* Maxpool */
    assign Maxpool_en = (cs == PASS_PPU && count_PPU != 3'd0 && maxpool);
    assign Maxpool_init = (cs == PASS_PPU && count_PPU == 3'd1 && maxpool);
    assign relu_sel = (cs == STORE_OFMAP && maxpool);


/* Multiplier */
    /* mul1 */
    always_comb  begin // mul1_src1
        case (cs)
            IDLE: mul1_src1 = E;
            LOAD_IFMAP: mul1_src1 = EF;
            LOAD_FILT: mul1_src1 = 'd9;
            SEND_PE_CONFIG: mul1_src1 = valid_e;
            SEND_IFMAP, SEND_AIFMAP: mul1_src1 = F;
            SEND_FILT, SEND_AFILT: mul1_src1 = 'd9;
            SEND_IPSUM, SEND_AIPSUM, STORE_OPSUM, STORE_OFMAP: mul1_src1 = count_e;
            PASS_PPU: begin
                if (maxpool) begin
                    case(count_PPU)
                        3'd0: mul1_src1 = {count_e,1'b0};
                        3'd1: mul1_src1 = {count_e,1'b0};
                        3'd2: mul1_src1 = {count_e,1'b1};
                        3'd3: mul1_src1 = {count_e,1'b1};
                        default: mul1_src1 = 'd0;
                    endcase
                end
                else mul1_src1 = count_e;
            end
            SEND_OFMAP: mul1_src1 = (maxpool)? EF[11:2]:EF;
            default: mul1_src1 = 'd0;
        endcase
    end

    always_comb  begin // mul1_src2
        case (cs)
            IDLE: mul1_src2 = F;
            LOAD_IFMAP: mul1_src2 = count_C + count_c;
            LOAD_FILT: mul1_src2 = count_C;
            SEND_OFMAP: mul1_src2 = count_M + count_m + count_PP_filt;
            SEND_PE_CONFIG: mul1_src2 = F;
            SEND_IFMAP, SEND_AIFMAP: mul1_src2 = count_h - {3'd0, H_top_padding};
            SEND_FILT, SEND_AFILT: mul1_src2 = count_PP_filt;
            SEND_IPSUM, SEND_AIPSUM, STORE_OPSUM, PASS_PPU: mul1_src2 = F;
            STORE_OFMAP: mul1_src2 = (maxpool)? {1'b0,F[7:1]}:F;
            default: mul1_src2 = 'd0;
        endcase
    end

    assign mul1_res = mul1_src1 * mul1_src2;

    /* mul2 */
    always_comb  begin // mul2_src1
        case (cs)
            IDLE: mul2_src1 = 'd9;
            LOAD_IFMAP: mul2_src1 = F;
            LOAD_FILT: mul2_src1 = C9;
            SEND_IPSUM, SEND_AIPSUM, STORE_OPSUM, PASS_PPU: mul2_src1 = eF;
            STORE_OFMAP: mul2_src1 = (maxpool)? eF[11:2]:eF;
            SEND_OFMAP: mul2_src1 = (maxpool)? {1'b0,F[7:1]}:F;
            default: mul2_src1 = 'd0;
        endcase
    end
    always_comb  begin // mul2_src2
        case (cs)
            IDLE: mul2_src2 = (C < 4)? 'd4: C;
            LOAD_IFMAP: mul2_src2 = count_H;
            LOAD_FILT: mul2_src2 = count_M + count_m + count_PP_filt;
            SEND_IPSUM, SEND_AIPSUM, STORE_OPSUM, PASS_PPU, STORE_OFMAP: mul2_src2 = count_m + count_PP_filt;
            SEND_OFMAP: mul2_src2 = (maxpool)? {1'b0,count_E[7:1]}:count_E;
            default: mul2_src2 = 'd0;
        endcase
    end
    assign mul2_res = mul2_src1 * mul2_src2;

    /* mul3 */
    always_comb  begin // mul3_src1
        case (cs)
            LOAD_IFMAP: mul3_src1 = F;
            LOAD_FILT: mul3_src1 = 'd9;
            SEND_OFMAP: mul3_src1 = (maxpool)? eF[11:2]:eF;
            default: mul3_src1 = 'd0;
        endcase
    end

    assign H_top_padding = (count_H == 8'd0);
    assign H_down_pdding = count_H_reset;
    always_comb  begin // mul3_src2
        case (cs)
            LOAD_IFMAP: mul3_src2 = h_max - {3'd0, H_top_padding} - {3'd0, H_down_pdding};
            LOAD_FILT: mul3_src2 =  count_PP_filt;
            SEND_OFMAP: mul3_src2 = count_m + count_PP_filt;
            default: mul3_src2 = 'd0;
        endcase
    end
    assign mul3_res = mul3_src1 * mul3_src2;

/* Mul Register */

always_ff @( posedge clk ) begin
    if (rst) begin
        C9 <= 'd0;
        eF <= 'd0;
        EF <= 'd0;
    end
    else begin
        if (cs == IDLE) begin
            EF <= mul1_res;
            C9 <= mul2_res;
        end
        else if (cs == SEND_PE_CONFIG) eF <= mul1_res;
    end
end


endmodule

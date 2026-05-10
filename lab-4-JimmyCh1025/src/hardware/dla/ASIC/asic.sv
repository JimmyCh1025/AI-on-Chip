`include "AXI_define.svh"
`include "ASIC.svh"

module asic(
    input clk,
    input rst,
    input asic_en,
    input maxpool_i,
    input relu_i,
    input operation_mode_i,
    input [5:0] scaling_factor_i,

    /* mapping parameters */
    input [9:0] m_i, // number of ofmap channels stored in GLB
    input [3:0] e_i, // width of the PE sets
    input [2:0] p_i, // number of filters processed by a PE set
    input [2:0] q_i, // number of channels processed by a PE
    input [2:0] r_i, // number of PE sets that process different channels in the PE arrays
    input [2:0] t_i, // number of PE sets that process different filters in the PE arrays

    /* shape parameters */
    input [9:0] C_i,
    input [9:0] M_i,
    input [7:0] W_i,
    input [7:0] H_i,

    /* DRAM config */
    input [`AXI_ADDR_BITS-1:0] ifmap_addr_i,
    input [`AXI_ADDR_BITS-1:0] filter_addr_i,
    input [`AXI_ADDR_BITS-1:0] bias_addr_i,
    input [`AXI_ADDR_BITS-1:0] ofmap_addr_i,

    // staring address in GLB (Note: GLB_ifmap_addr = 0)
    input [`GLB_ADDR_BITS-1:0] GLB_filter_addr_i,
    input [`GLB_ADDR_BITS-1:0] GLB_bias_addr_i,
    input [`GLB_ADDR_BITS-1:0] GLB_opsum_addr_i,

    /* GLB */
    output logic GLB_EN,
    output logic GLB_WEB,
    output logic GLB_MODE,
    output logic [`GLB_ADDR_BITS-1:0] GLB_A,
    output logic [`DATA_BITS-1:0] GLB_DI,
    input [`DATA_BITS-1:0] GLB_DO,
    output logic GLB_mux,

    /* DMA */
    output logic DMA_en,
    output logic [1:0] DMA_mode,
    output logic [`AXI_ADDR_BITS-1:0] DMA_DRAM_ADDR,
    output logic [`GLB_ADDR_BITS-1:0] DMA_GLB_ADDR,
    output logic [`GLB_ADDR_BITS-1:0] DMA_len,
    output logic [1:0] DMA_byte_bias,
    input DMA_done,

    output logic asic_interrupt
);

/*******************************************
    PE - array
********************************************/
logic [`DATA_BITS-1:0] GLB_data_out_PEarray;
logic [`DATA_BITS-1:0] GLB_data_in_PEarray;

logic set_XID;
logic [`XID_BITS-1:0] ifmap_XID_scan_in;
logic [`XID_BITS-1:0] filter_XID_scan_in;
logic [`XID_BITS-1:0] ipsum_XID_scan_in;
logic [`XID_BITS-1:0] opsum_XID_scan_in;
logic set_YID;
logic [`YID_BITS-1:0] ifmap_YID_scan_in;
logic [`YID_BITS-1:0] filter_YID_scan_in;
logic [`YID_BITS-1:0] ipsum_YID_scan_in;
logic [`YID_BITS-1:0] opsum_YID_scan_in;
logic set_LN;
logic [`PE_ARRAY_H-2:0] LN_config_in;

logic [`PE_ARRAY_H*`PE_ARRAY_W-1:0] PE_en;
logic [10:0] PE_config;

logic PEA_ifmap_valid;
logic PEA_ifmap_ready;
logic [`XID_BITS-1:0] ifmap_tag_X;
logic [`YID_BITS-1:0] ifmap_tag_Y;

logic PEA_filter_valid;
logic PEA_filter_ready;
logic [`XID_BITS-1:0] filter_tag_X;
logic [`YID_BITS-1:0] filter_tag_Y;

logic PEA_ipsum_valid;
logic PEA_ipsum_ready;
logic [`XID_BITS-1:0] ipsum_tag_X;
logic [`YID_BITS-1:0] ipsum_tag_Y;

logic PEA_opsum_valid;
logic PEA_opsum_ready;
logic [`XID_BITS-1:0] opsum_tag_X;
logic [`YID_BITS-1:0] opsum_tag_Y;

logic GLB_DI_select;
logic GLB_DO_select;

PE_array PE_array(
    .clk(clk),
    .rst(rst),
    /* Scan Chain */
    .set_XID(set_XID),
    .ifmap_XID_scan_in(ifmap_XID_scan_in),
    .filter_XID_scan_in(filter_XID_scan_in),
    .ipsum_XID_scan_in(ipsum_XID_scan_in),
    .opsum_XID_scan_in(opsum_XID_scan_in),
    .set_YID(set_YID),
    .ifmap_YID_scan_in(ifmap_YID_scan_in),
    .filter_YID_scan_in(filter_YID_scan_in),
    .ipsum_YID_scan_in(ipsum_YID_scan_in),
    .opsum_YID_scan_in(opsum_YID_scan_in),
    .set_LN(set_LN),
    .LN_config_in(LN_config_in),

    /* Controller */
    .PE_en(PE_en),
    .PE_config(PE_config),
    .ifmap_tag_X(ifmap_tag_X),
    .ifmap_tag_Y(ifmap_tag_Y),
    .filter_tag_X(filter_tag_X),
    .filter_tag_Y(filter_tag_Y),
    .ipsum_tag_X(ipsum_tag_X),
    .ipsum_tag_Y(ipsum_tag_Y),
    .opsum_tag_X(opsum_tag_X),
    .opsum_tag_Y(opsum_tag_Y),

    /* GLB */
    .GLB_ifmap_valid(PEA_ifmap_valid),
    .GLB_ifmap_ready(PEA_ifmap_ready),
    .GLB_filter_valid(PEA_filter_valid),
    .GLB_filter_ready(PEA_filter_ready),
    .GLB_ipsum_valid(PEA_ipsum_valid),
    .GLB_ipsum_ready(PEA_ipsum_ready),
    .GLB_data_in(GLB_data_in_PEarray),

    .GLB_opsum_valid(PEA_opsum_valid),
    .GLB_opsum_ready(PEA_opsum_ready),
    .GLB_data_out(GLB_data_out_PEarray)
);

logic [7:0] ppu_data_out;
logic relu_sel, comp_en, comp_init;

PPU PPU (
    .clk(clk),
    .rst(rst),
    .data_in(GLB_DO),
    .scaling_factor(scaling_factor_i),
    .maxpool_en(comp_en),
    .maxpool_init(comp_init),
    .relu_sel(relu_sel),
    .relu_en(relu_i),
    .data_out(ppu_data_out)
);

always_comb begin
    /* GLB DI mux */
    GLB_DI = (GLB_DI_select == `GLB_DO_PSUM)?GLB_data_out_PEarray:{24'd0,ppu_data_out};
    /* GLB DO mux */
    GLB_data_in_PEarray = (GLB_DO_select == `WITH_PAD)?32'h80808080:GLB_DO;
end

/*******************************************
    ASIC controller
********************************************/

asic_controller asic_controller_0(
    .clk(clk),
    .rst(rst),
    .asic_en(asic_en),
    .asic_done(asic_interrupt),
    /* MMIO */
    .ifmap_addr(ifmap_addr_i),
    .filter_addr(filter_addr_i),
    .bias_addr(bias_addr_i),
    .ofmap_addr(ofmap_addr_i),
    .GLB_filter_addr(GLB_filter_addr_i),
    .GLB_bias_addr(GLB_bias_addr_i),
    .GLB_opsum_addr(GLB_opsum_addr_i),
    /* Layer Info */
    .maxpool(maxpool_i),
    /* mapping parameters */
    .m(m_i),
    .e(e_i),
    .p(p_i),
    .q(q_i),
    .r(r_i),
    .t(t_i),
    /* shape parameters */
    .C(C_i),
    .M(M_i),
    .W(W_i),
    .H(H_i),
    /* DMA */
    .DMA_en(DMA_en),
    .DMA_mode(DMA_mode),
    .DMA_DRAM_ADDR(DMA_DRAM_ADDR),
    .DMA_GLB_ADDR(DMA_GLB_ADDR),
    .DMA_len(DMA_len),
    .DMA_byte_bias(DMA_byte_bias),
    .DMA_done(DMA_done),
    /* ID config */
    .set_XID(set_XID),
    .ifmap_XID_scan_in(ifmap_XID_scan_in),
    .filter_XID_scan_in(filter_XID_scan_in),
    .ipsum_XID_scan_in(ipsum_XID_scan_in),
    .opsum_XID_scan_in(opsum_XID_scan_in),
    .set_YID(set_YID),
    .ifmap_YID_scan_in(ifmap_YID_scan_in),
    .filter_YID_scan_in(filter_YID_scan_in),
    .ipsum_YID_scan_in(ipsum_YID_scan_in),
    .opsum_YID_scan_in(opsum_YID_scan_in),
    .set_LN(set_LN),
    .LN_config_in(LN_config_in),

    /* PE_Array */
    .PE_en(PE_en),
    .PE_config(PE_config),

    .PEA_ifmap_valid(PEA_ifmap_valid),
    .PEA_ifmap_ready(PEA_ifmap_ready),
    .ifmap_tag_X(ifmap_tag_X),
    .ifmap_tag_Y(ifmap_tag_Y),


    .PEA_filter_valid(PEA_filter_valid),
    .PEA_filter_ready(PEA_filter_ready),
    .filter_tag_X(filter_tag_X),
    .filter_tag_Y(filter_tag_Y),

    .PEA_ipsum_valid(PEA_ipsum_valid),
    .PEA_ipsum_ready(PEA_ipsum_ready),
    .ipsum_tag_X(ipsum_tag_X),
    .ipsum_tag_Y(ipsum_tag_Y),

    .PEA_opsum_valid(PEA_opsum_valid),
    .PEA_opsum_ready(PEA_opsum_ready),
    .opsum_tag_X(opsum_tag_X),
    .opsum_tag_Y(opsum_tag_Y),

    /* GLB */
    .GLB_EN(GLB_EN),
    .GLB_WEB(GLB_WEB),
    .GLB_MODE(GLB_MODE),
    .GLB_A(GLB_A),
    .GLB_mux(GLB_mux),
    .GLB_DI_select(GLB_DI_select),
    .GLB_DO_select(GLB_DO_select),

    /* PPU */
    .Maxpool_en(comp_en),
    .Maxpool_init(comp_init),
    .relu_sel(relu_sel)
);


endmodule


module PE #(
    parameter IFMAP_SIZE = 8,
    parameter FILTER_SIZE = 8,
    parameter PSUM_SIZE = 32,
    parameter IFMAP_SPAD_LEN = 12,
    parameter FILTER_SPAD_LEN = 48,
    parameter OFMAP_SPAD_LEN = 4,
    parameter IFMAP_INDEX_BIT = 4,
    parameter FILTER_INDEX_BIT = 6,
    parameter OFMAP_INDEX_BIT = 2,
    parameter OFMAP_COL_BIT = 5
)(
    input clk,
    input rst,
    input PE_en,
    input [10:0] i_config,
    input [31:0] ifmap,
    input [31:0] filter,
    input [31:0] ipsum,
    input ifmap_valid,
    input filter_valid,
    input ipsum_valid,
    input opsum_ready,
    output logic [31:0] opsum,
    output logic ifmap_ready,
    output logic filter_ready,
    output logic ipsum_ready,
    output logic opsum_valid
);

// *** Constraint - input channel must be a multiple of 4 ***

/*
|========================================= config spec =========================================|
| ordering - channel first N -> H -> W -> C                                                     |
| config - {operation_mode, output_ch[1:0], ofmap_col[4:0], input_ch[1:0]}                          |
|===============================================================================================|

the configs you get are CONFIG- 1 EX: input_ch = 2'd3, but actually input channel is 4.
*/

/*
|============== ipsum ordering ==============|
|        ofmap_col -> ofmap_channel          |
|============================================|
*/

logic [IFMAP_SIZE-1:0] ifmap_spad [0:IFMAP_SPAD_LEN-1];
logic [FILTER_SIZE-1:0] filter_spad [0:FILTER_SPAD_LEN-1];
logic [PSUM_SIZE-1:0] ofmap_spad [0:OFMAP_SPAD_LEN-1];

logic [1:0] count_filt_col;
logic [1:0] input_ch, count_input_ch;
logic [OFMAP_COL_BIT-1:0] ofmap_col, count_ofmap_col;
logic [OFMAP_INDEX_BIT-1:0] ofmap_ch, count_ofmap_ch;
logic operation_mode;
logic load_ipsum;

logic [IFMAP_INDEX_BIT-1:0] ifmap_index;
logic [FILTER_INDEX_BIT-1:0] filter_index;
logic [OFMAP_INDEX_BIT-1:0] ofmap_index;


logic [FILTER_INDEX_BIT-1:0] select_index;
logic [FILTER_INDEX_BIT-1:0] index_0, index_1, index_2, index_3;

logic signed [IFMAP_SIZE+FILTER_SIZE-1:0] mul_res;
logic [PSUM_SIZE-1:0] add_res, add_src1, add_src2;

integer i;

parameter FILT_COL = 2'd2;

typedef enum logic [3:0] {
    IDLE,
    LOAD_FILT,
    LOAD_IFMAP,
    COMPUTE,
    ADD_IPSUM,
    STORE_OPSUM,
    LOAD_IFMAP_REUSE
} state_t;
state_t cs, ns;

parameter IPSUM_N = 1'd0,
          IPSUM_Y = 1'd1;

assign ifmap_index = {count_filt_col, count_input_ch};
assign filter_index = {count_filt_col, count_ofmap_ch, count_input_ch};
assign ofmap_index = count_ofmap_ch;

assign select_index = (cs == LOAD_FILT)? filter_index : {{(FILTER_INDEX_BIT-IFMAP_INDEX_BIT){1'b0}}, ifmap_index};
assign index_0 = select_index;
assign index_1 = select_index + 'd1;
assign index_2 = select_index + 'd2;
assign index_3 = select_index + 'd3;

assign mul_res = $signed(ifmap_spad[ifmap_index]) * $signed(filter_spad[filter_index]);
assign add_src1 = ofmap_spad[ofmap_index];
assign add_src2 = (cs == COMPUTE)? {{(PSUM_SIZE-IFMAP_SIZE-FILTER_SIZE){mul_res[IFMAP_SIZE+FILTER_SIZE-1]}}, mul_res}: ipsum;
assign add_res = add_src1 + add_src2;

always_ff @( posedge clk ) begin // FSM_seq
    if (rst) cs <= IDLE;
    else cs <= ns;
end

always_comb begin // FSM_comb
    case (cs)
        IDLE: begin
            if (PE_en) ns = LOAD_FILT;
            else ns = IDLE;
        end
        LOAD_FILT: begin
            if (filter_valid && count_filt_col == FILT_COL && count_ofmap_ch == ofmap_ch) ns = LOAD_IFMAP;
            else ns = LOAD_FILT;
        end
        LOAD_IFMAP: begin
            if (ifmap_valid && count_filt_col == FILT_COL) ns = COMPUTE;
            else ns = LOAD_IFMAP;
        end
        COMPUTE: begin
            if (count_filt_col == FILT_COL && count_input_ch == input_ch && count_ofmap_ch == ofmap_ch) begin
                ns = ADD_IPSUM;
            end
            else ns = COMPUTE;
        end
        ADD_IPSUM: begin
            if (ipsum_valid && count_ofmap_ch == ofmap_ch) ns = STORE_OPSUM;
            else ns = ADD_IPSUM;
        end
        STORE_OPSUM: begin
            if (opsum_ready && count_ofmap_ch == ofmap_ch) begin
                if (count_ofmap_col == ofmap_col) ns = IDLE;
                else ns = LOAD_IFMAP_REUSE;
            end
            else ns = STORE_OPSUM;
        end
        LOAD_IFMAP_REUSE: begin
            if (ifmap_valid) ns = COMPUTE;
            else ns = LOAD_IFMAP_REUSE;
        end
        default: ns = IDLE;
    endcase
end

always_ff @( posedge clk ) begin // SETUP
    if (rst) begin
        input_ch <= 'd0;
        ofmap_col <= 'd0;
        ofmap_ch <= 'd0;
        operation_mode <= 1'b0;
    end
    else begin
        if (cs == IDLE && PE_en) begin
            input_ch <= i_config[1:0];
            ofmap_col <= i_config[OFMAP_COL_BIT+1:2];
            ofmap_ch <= i_config[OFMAP_COL_BIT+OFMAP_INDEX_BIT+1:OFMAP_COL_BIT+2];
            operation_mode <= i_config[OFMAP_COL_BIT+OFMAP_INDEX_BIT+2];
        end
    end
end

// ==================== PE counter ====================
always_ff @( posedge clk ) begin // count_filt_col
    if (rst) count_filt_col <= 'd0;
    else begin
        case (cs)
            LOAD_FILT: begin
                if (filter_valid) begin
                    if (count_filt_col == FILT_COL) count_filt_col <= 'd0;
                    else count_filt_col <= count_filt_col + 'd1;
                end
            end
            LOAD_IFMAP: begin
                if (ifmap_valid) begin
                    if (count_filt_col == FILT_COL) count_filt_col <= 'd0;
                    else count_filt_col <= count_filt_col + 'd1;
                end
            end
            COMPUTE: begin
                if (count_input_ch == input_ch) begin
                    if (count_filt_col == FILT_COL) count_filt_col <= 'd0;
                    else count_filt_col <= count_filt_col + 'd1;
                end
            end
            default: count_filt_col <= count_filt_col;
        endcase
    end
end

always_ff @( posedge clk ) begin // count_ofmap_ch
    if (rst) count_ofmap_ch <= 2'd0;
    else begin
        case (cs)
            LOAD_FILT: begin
                if (filter_valid && count_filt_col == FILT_COL) begin
                    if (count_ofmap_ch == ofmap_ch) count_ofmap_ch <= 'd0;
                    else count_ofmap_ch <= count_ofmap_ch + 'd1;
                end
            end
            COMPUTE: begin
                if (count_input_ch == input_ch && count_filt_col == FILT_COL) begin
                    if (count_ofmap_ch == ofmap_ch) count_ofmap_ch <= 'd0;
                    else count_ofmap_ch <= count_ofmap_ch + 'd1;
                end
            end
            ADD_IPSUM: begin
                if (ipsum_valid) begin
                    if (count_ofmap_ch == ofmap_ch) count_ofmap_ch <= 'd0;
                    else count_ofmap_ch <= count_ofmap_ch + 'd1;
                end
            end
            STORE_OPSUM: begin
                if (opsum_ready) begin
                    if (count_ofmap_ch == ofmap_ch) count_ofmap_ch <= 'd0;
                    else count_ofmap_ch <= count_ofmap_ch + 'd1;
                end
            end
            default: count_ofmap_ch <= count_ofmap_ch;
        endcase
    end
end

always_ff @( posedge clk ) begin // count_input_ch
    if (rst) count_input_ch <= 'd0;
    else begin
        if (cs == COMPUTE) begin
            if (count_input_ch == input_ch) count_input_ch <= 'd0;
            else count_input_ch <= count_input_ch + 'd1;
        end
    end
end

always_ff @( posedge clk ) begin // count_ofmap_col
    if (rst) count_ofmap_col <= 'd0;
    else begin
        if ((cs == STORE_OPSUM) && (count_ofmap_ch == ofmap_ch) && opsum_ready) begin
            if (count_ofmap_col == ofmap_col) count_ofmap_col <= 'd0;
            else count_ofmap_col <= count_ofmap_col + 'd1;
        end
    end
end

// ==================== PE spad ====================
always_ff @( posedge clk ) begin // filter_spad
    integer i;
    if (rst) for (i = 0; i < 24; i = i + 1) filter_spad[i] <= 8'd0;
    else begin
        if (cs == LOAD_FILT && filter_valid) begin
            filter_spad[index_0] <= filter[7:0];
            filter_spad[index_1] <= filter[15:8];
            filter_spad[index_2] <= filter[23:16];
            filter_spad[index_3] <= filter[31:24];
        end
    end
end

always_ff @( posedge clk ) begin // ifmap_spad
    integer i;
    if (rst) for (i = 0; i < 12; i = i + 1) ifmap_spad[i] <= 8'd0;
    else begin
        if (cs == LOAD_IFMAP && ifmap_valid) begin
            ifmap_spad[index_0[3:0]] <= {!ifmap[7], ifmap[6:0]};
            ifmap_spad[index_1[3:0]] <= (input_ch < 2'd1)? 8'd0 :{!ifmap[15], ifmap[14:8]};
            ifmap_spad[index_2[3:0]] <= (input_ch < 2'd2)? 8'd0 :{!ifmap[23], ifmap[22:16]};
            ifmap_spad[index_3[3:0]] <= (input_ch < 2'd3)? 8'd0 :{!ifmap[31], ifmap[30:24]};
        end
        else if (cs == LOAD_IFMAP_REUSE && ifmap_valid) begin
            ifmap_spad[0] <= ifmap_spad[4];
            ifmap_spad[1] <= ifmap_spad[5];
            ifmap_spad[2] <= ifmap_spad[6];
            ifmap_spad[3] <= ifmap_spad[7];
            ifmap_spad[4] <= ifmap_spad[8];
            ifmap_spad[5] <= ifmap_spad[9];
            ifmap_spad[6] <= ifmap_spad[10];
            ifmap_spad[7] <= ifmap_spad[11];
            ifmap_spad[8] <= {!ifmap[7], ifmap[6:0]};
            ifmap_spad[9] <= (input_ch < 2'd1)? 8'd0:{!ifmap[15], ifmap[14:8]};
            ifmap_spad[10] <= (input_ch < 2'd2)? 8'd0:{!ifmap[23], ifmap[22:16]};
            ifmap_spad[11] <= (input_ch < 2'd3)? 8'd0:{!ifmap[31], ifmap[30:24]};
        end
    end
end

always_ff @( posedge clk ) begin // ofmap_spad
    if (rst) begin
        for (i = 0; i < OFMAP_SPAD_LEN; i++) ofmap_spad[i] <= 'd0;
    end
    else begin
        if (cs == IDLE || cs == LOAD_IFMAP_REUSE) for (i = 0; i < OFMAP_SPAD_LEN; i++) ofmap_spad[i] <= 'd0;
        else if (cs == COMPUTE) ofmap_spad[ofmap_index] <= add_res;
        else if (cs == ADD_IPSUM && ipsum_valid) begin
            ofmap_spad[ofmap_index] <= add_res;
        end
    end
end

assign opsum = (cs == STORE_OPSUM)? ofmap_spad[ofmap_index] : 32'd0;
assign filter_ready = (cs == LOAD_FILT)? 1'b1 : 1'b0;
assign ifmap_ready = (cs == LOAD_IFMAP || cs == LOAD_IFMAP_REUSE)? 1'b1 : 1'b0;
assign ipsum_ready = (cs == ADD_IPSUM)? 1'b1 : 1'b0;
assign opsum_valid = (cs == STORE_OPSUM)? 1'b1 : 1'b0;

endmodule

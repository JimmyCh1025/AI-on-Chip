`include "define.svh"
`define MAX_CH 4

module PE (
    input clk,
    input rst,
    input PE_en,
    input [`CONFIG_SIZE-1:0] i_config,
    input [`DATA_BITS-1:0] ifmap,
    input [`DATA_BITS-1:0] filter,
    input [`DATA_BITS-1:0] ipsum,
    input ifmap_valid,
    input filter_valid,
    input ipsum_valid,
    input opsum_ready,
    output logic [`DATA_BITS-1:0] opsum,
    output logic ifmap_ready,
    output logic filter_ready,
    output logic ipsum_ready,
    output logic opsum_valid
);

    typedef enum logic [2:0] {
        S_LOAD_CONFIG, // receive config   
        S_LOAD_FILT,   // receive Filter
        S_LOAD_IFMAP,  // receive Ifmap
        S_SHIFT_IFMAP, // shift Ifmap
        S_LOAD_IPSUM,  // receive Ipsum 
        S_ACCUMULATE,  // accumulate psum += ifmap * filter  
        S_STORE_OPSUM, // send Opsum
        S_DONE         // done
    } state_t;

    // State
    state_t curr_state, next_state;

    // Activation of PE
    logic [1:0] config_ich;      // q-1 (Input Channels)
    logic [4:0] config_col;      // F-1 (Output Columns)
    logic [1:0] config_och;      // p-1 (Output Channels)
    logic       config_mode;     // mode(mode 0 = CONV, mode 1 = FC Layer)

    // Counter 
    logic [5:0] counter;
    logic [4:0] kernel_counter;

    // Index 
    logic [1:0] filter_idx; // output filter index
    logic [1:0] channel_idx; // input channel index
    logic [1:0] kernel_idx; // kernel index

    // Data    
    logic signed [`IFMAP_SIZE-1:0]  ifmap_spad  [`IFMAP_SPAD_LEN-1:0];  // 12 bytes IFMAP SPAD , 8 bits ifmap, width = 3, input channel = 4
    logic signed [`FILTER_SIZE-1:0] filter_spad [`FILTER_SPAD_LEN-1:0]; // 48 bytes FILTER SPAD, 8 bits filter, width = 3, input channel = 4, output channel = 4
    logic signed [`PSUM_SIZE-1:0]   opsum_spad  [`OFMAP_SPAD_LEN-1:0];  // 16 bytes PSUM SPAD, 32 bits psum, output channel = 4

    // output signal 
    assign filter_ready = (curr_state == S_LOAD_FILT);
    assign ifmap_ready  = ((curr_state == S_LOAD_IFMAP) || (curr_state == S_SHIFT_IFMAP));
    assign ipsum_ready  = (curr_state == S_LOAD_IPSUM);
    assign opsum_valid  = (curr_state == S_STORE_OPSUM);
    assign opsum        = opsum_spad[counter[1:0]];

    // counter 
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            counter        <= 0;
            kernel_counter <= 0;
        end else begin
            case (curr_state) 

                S_LOAD_FILT : begin
                    // if filter valid, receive filter data and counter + MAX Channel
                    if (filter_valid) begin
                        if (counter == (`FILTER_SPAD_LEN - `MAX_CH))
                            counter <= 0;
                        else
                            counter <= counter + `MAX_CH; 
                    end else begin
                        counter <= counter;
                    end
                end

                S_LOAD_IFMAP : begin
                    // if ifmap valid, receive ifmap data and counter + MAX Channel
                    if (ifmap_valid) begin
                        if (counter == (`IFMAP_SPAD_LEN - `MAX_CH))
                            counter <= 0;
                        else
                            counter <= counter + `MAX_CH; 
                    end else begin
                        counter <= counter;
                    end
                end
                
                S_LOAD_IPSUM : begin
                    // if ipsum valid, receive ipsum data and counter + 1
                    if (ipsum_valid) begin
                        if (counter[1:0] == config_och) 
                            counter <= 0;
                        else
                            counter <= counter + 1;
                    end else begin
                        counter <= counter;
                    end
                end

                S_STORE_OPSUM : begin
                    // if opsum ready, send opsum data and counter + 1
                    if (opsum_ready) begin
                        if (counter[1:0] == config_och) begin
                            counter <= 0;
                            if (opsum_valid) begin
                                kernel_counter <= kernel_counter + 1;
                            end
                        end else begin
                            counter <= counter + 1;
                        end
                    end else begin
                        counter <= counter;
                    end
                end

                default : begin
                    counter <= 0;
                end
            endcase
        end
    end

    // idx of accumulator
    always_ff @(posedge clk) begin
        case (curr_state) 
            S_ACCUMULATE : begin
                // filter [output channel][kernel][input channel]
                // check filter finish
                filter_idx  <= (kernel_idx == (`FILT_S - 1))? ((filter_idx == config_och)? 0 : filter_idx + 1) : filter_idx;
                // if one channel filter finish, then go to next filter
                channel_idx <= (filter_idx == config_och) && (kernel_idx == (`FILT_S - 1))? channel_idx + 1 : channel_idx;
                // check kernel index(0~2)
                kernel_idx  <= (kernel_idx == (`FILT_S - 1))? 0 : kernel_idx + 1;
            end

            default : begin
                filter_idx  <= 0;
                channel_idx <= 0;
                kernel_idx  <= 0;
            end

        endcase
    end



    // SPAD 
    always_ff @(posedge clk) begin
        case (curr_state) 
            S_LOAD_FILT : begin
                // `define FILTER_SPAD_LEN 48 => 6 bits
                for (int i = 0 ; i < 4 ; ++i) 
                    // i*FILTER_SIZE : i*FILTER_SIZE-1 => 8 bits 
                    filter_spad[counter + i[5:0]] <= filter[i*`FILTER_SIZE +: `FILTER_SIZE];
            end

            S_LOAD_IFMAP : begin
                // `define IFMAP_SPAD_LEN 12 => 4 bits
                for (int i = 0 ; i < 4 ; ++i) 
                    // ~ifmap[(i + 1) * `IFMAP_SIZE - 1] => [ ReLU(+128, range[0, 255]) => Ifmap(-128, range[-128, 127])]
                    ifmap_spad[counter[3:0] + i[3:0]] <= {~ifmap[(i + 1) * `IFMAP_SIZE - 1], ifmap[i * `IFMAP_SIZE +: `IFMAP_SIZE-1]};
            end

            S_SHIFT_IFMAP : begin
                if (ifmap_valid) begin
                    for (int i = 0 ; i < 4 ; ++i) begin
                        // sliding window shift, 4 channel same time => col 1 -> col 0
                        ifmap_spad[i    ] <= ifmap_spad[4 + i];
                        // next ifmap => col 2 -> col 1
                        ifmap_spad[4 + i] <= ifmap_spad[8 + i];
                        // col 3 -> col 2
                        ifmap_spad[8 + i] <= {~ifmap[(i + 1) * `IFMAP_SIZE - 1], ifmap[i * `IFMAP_SIZE +: `IFMAP_SIZE-1]};
                    end
                end
            end

            S_LOAD_IPSUM : begin
                // `define OFMAP_SPAD_LEN 4 => 2 bits
                if (ipsum_valid) begin
                    opsum_spad[counter[1:0]] <= ipsum;
                end
            end

            S_ACCUMULATE : begin
                // filter spad => filter index 12(kernel size(3) * input channel(4)) => no.x filter, next convolution point(4) => no.x col, next channel(1) => no.x channel
                // ifmap spad  => next convolution point(4), next channel(1)
                opsum_spad[filter_idx] <= opsum_spad[filter_idx] + filter_spad[(6'(filter_idx) * 12) + (6'(kernel_idx) << 2) + 6'(channel_idx)] * ifmap_spad[(4'(kernel_idx) << 2) | 4'(channel_idx)];
            end


            default : begin

            end


        endcase
    end

    // set config
    always_ff @(posedge clk) begin
        case (curr_state) 
            S_LOAD_CONFIG : begin
                {config_mode, config_och, config_col, config_ich} <= i_config;
            end

            default : begin
                {config_mode, config_och, config_col, config_ich} <= {config_mode, config_och, config_col, config_ich};
            end
        endcase
    end


    // set current state to next state
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            curr_state <= S_LOAD_CONFIG;
        end else begin
            curr_state <= next_state;
        end
    end

    // state machine
    always_comb begin
        next_state = S_LOAD_CONFIG;
        case (curr_state) 
            S_LOAD_CONFIG : begin
                if (PE_en) begin
                    next_state = S_LOAD_FILT;
                end else begin
                    next_state = S_LOAD_CONFIG;
                end
            end

            S_LOAD_FILT : begin
                if ((counter == `FILTER_SPAD_LEN - `MAX_CH) && filter_valid) begin
                    next_state = S_LOAD_IFMAP;
                end else begin
                    next_state = S_LOAD_FILT;
                end
            end

            S_LOAD_IFMAP : begin
                if ((counter == `IFMAP_SPAD_LEN - `MAX_CH) && ifmap_valid) begin
                    next_state = S_LOAD_IPSUM;
                end else begin
                    next_state = S_LOAD_IFMAP;
                end
            end

            S_SHIFT_IFMAP : begin
                if (ifmap_valid) begin
                    next_state = S_LOAD_IPSUM;
                end else begin
                    next_state = S_SHIFT_IFMAP;
                end
            end

            S_LOAD_IPSUM : begin
                if ((counter[1:0] == config_och) && ipsum_valid) begin
                    next_state = S_ACCUMULATE;
                end else begin
                    next_state = S_LOAD_IPSUM;
                end
            end

            S_ACCUMULATE : begin
                if ((kernel_idx == (`FILT_S - 1)) && (channel_idx == config_ich) && (filter_idx == config_och)) begin
                    next_state = S_STORE_OPSUM;
                end else begin
                    next_state = S_ACCUMULATE;
                end
            end

            S_STORE_OPSUM : begin
                if ((counter[1:0] == config_och) && opsum_ready) begin
                    if (kernel_counter == config_col) begin
                        next_state = S_DONE;
                    end else begin
                        next_state = S_SHIFT_IFMAP;
                    end
                end else begin
                    next_state = S_STORE_OPSUM;
                end
            end

            S_DONE : begin
                next_state = S_DONE;
            end

            default : begin
                next_state = S_LOAD_CONFIG;
            end

        endcase
    end

endmodule

#include <stdio.h>
#include <stdlib.h>

#include "driver_dla.h"
#include "runtime.h"

/*  //////////      NOTICE      //////////
    all parameter used to set DLA are send in by function argument
*/

void dla_stop() {
    // set disable
    reg_write(DLA_ENABLE_OFFSET, 0);
}

void create_dla_info_to_csv(const char* filename) {
    fprintf(stdout, "Creating dla info file: %s\n", filename);
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Create DLA info file failed.\n");
        return;
    }
    fprintf(file,
            "Operation,Cycles,Time(ns),Memory read,Memory "
            "write,m,e,p,q,r,t,PAD,U,R,S,C,M,W,H\n");
    fclose(file);
}

void dump_dla_info_to_csv(const char* filename, const char* operation_name,
                          // mapping parameter
                          uint32_t m, uint32_t e, uint32_t p, uint32_t q,
                          uint32_t r, uint32_t t,
                          // shape parameter
                          uint32_t PAD, uint32_t U, uint32_t R, uint32_t S,
                          uint32_t C, uint32_t M, uint32_t W, uint32_t H) {
    FILE* file = fopen(filename, "a");
    struct runtime_info info = get_dla_hal()->get_runtime_info();
    fprintf(file, "%s,", operation_name);  // Operation
    fprintf(file, "%10llu,", (unsigned long long)info.elapsed_cycle);  // Cycles
    fprintf(file, "%10llu,", (unsigned long long)info.elapsed_time);   // Time (ns)
    fprintf(file, "%10d,", info.memory_read);        // Memory read
    fprintf(file, "%10d,", info.memory_write);       // Memory write
    fprintf(file, "%d,%d,%d,%d,%d,%d,", m, e, p, q, r, t);
    fprintf(file, "%d,%d,%d,%d,%d,%d,%d,%d\n", PAD, U, R, S, C, M, W, H);
    fclose(file);
}

int qconv2d_relu_maxpool(
    uint8_t* input_in_DRAM, int8_t* filter_in_DRAM, uint8_t* opsum_in_DRAM,
    int32_t* bias, uint32_t ofmap_len, uint32_t ifmap_len, uint32_t filter_len,
    // mapping parameter
    uint32_t m, uint32_t e, uint32_t p, uint32_t q, uint32_t r, uint32_t t,
    // shape parameter
    uint32_t PAD, uint32_t U, uint32_t R, uint32_t S, uint32_t C, uint32_t M,
    uint32_t W, uint32_t H,
    uint32_t scale) {  // int32_t scale_factor: merge ifmap and weight and ofmap
    // scale bit-shift

#ifdef DLA_INFO
    dla_reset_runtime_info();
#endif
    
    // Padding | W | Padding  
    uint32_t F = (W + (2 * PAD) - S)/U + 1; 
    uint32_t glb_size = 64 * 1024; // 64KB
    uint32_t glb_ifmap_usage = 1 * (q * r) * (U * (e - 1) + R) * W * 1;
    uint32_t glb_filter_usage = (p * t) * (q * r) * R * S * 1;
    uint32_t glb_bias_usage = m * 4; // This reduces DRAM access and saves handshake time. 
    //uint32_t glb_opsum_usage = 1 * m * e * F * 4;
    
    m = (glb_size - glb_ifmap_usage - glb_filter_usage - glb_bias_usage) / (e * F * 4);
    m = 1 << (uint32_t)log2(m);
    m = (m <= M) ? m : M;


    // shape
    set_mapping_param(m, e, p, q, r, t);
    set_shape_param1(PAD, U, R, S, C, M);
    set_shape_param2(W, H, PAD);

    // DRAM address
    set_ifmap_addr(input_in_DRAM);
    set_filter_addr(filter_in_DRAM);
    set_bias_addr(bias);
    set_opsum_addr(opsum_in_DRAM);

    // GLB address
    set_glb_filter_addr(glb_ifmap_usage);
    set_glb_bias_addr(glb_ifmap_usage + glb_filter_usage);
    set_glb_ofmap_addr(glb_ifmap_usage + glb_filter_usage + glb_bias_usage);

    // activation lengths
    set_input_activation_len(ifmap_len);
    set_output_activation_len(ofmap_len);

    // set_enable(uint32_t scale_factor, bool maxpool, bool relu, bool operation) 
    set_enable(scale, true, true, 0);


    get_dla_hal()->wait_for_irq();
    dla_stop();
#ifdef DLA_INFO
    dump_dla_info_to_csv(DLA_INFO_CSV, "qconv2d_relu_maxpool", m, e, p, q, r, t,
                         PAD, U, R, S, C, M, W, H);
#endif
    return 0;
};

int qconv2d_relu(uint8_t* input_in_DRAM, int8_t* filter_in_DRAM,
                 uint8_t* opsum_in_DRAM, int32_t* bias, uint32_t ofmap_len,
                 uint32_t ifmap_len, uint32_t filter_len,
                 // mapping parameter
                 uint32_t m, uint32_t e, uint32_t p, uint32_t q, uint32_t r,
                 uint32_t t,
                 // shape parameter
                 uint32_t PAD, uint32_t U, uint32_t R, uint32_t S, uint32_t C,
                 uint32_t M, uint32_t W, uint32_t H,
                 uint32_t scale) {  // int32_t scale_factor: merge ifmap and
                                    // ofmap scale bit-shift
#ifdef DLA_INFO
    dla_reset_runtime_info();
#endif

    // Padding | W | Padding  
    uint32_t F = (W + (2 * PAD) - S)/U + 1; 
    uint32_t glb_size = 64 * 1024; // 64KB
    uint32_t glb_ifmap_usage = 1 * (q * r) * (U * (e - 1) + R) * W * 1;
    uint32_t glb_filter_usage = (p * t) * (q * r) * R * S * 1;
    uint32_t glb_bias_usage = m * 4; // This reduces DRAM access and saves handshake time. 
    //uint32_t glb_opsum_usage = 1 * m * e * F * 4;
    
    m = (glb_size - glb_ifmap_usage - glb_filter_usage - glb_bias_usage) / (e * F * 4);
    m = 1 << (uint32_t)log2(m);
    m = (m <= M) ? m : M;   


    // shape
    set_mapping_param(m, e, p, q, r, t);
    set_shape_param1(PAD, U, R, S, C, M);
    set_shape_param2(W, H, PAD);

    // DRAM address
    set_ifmap_addr(input_in_DRAM);
    set_filter_addr(filter_in_DRAM);
    set_bias_addr(bias);
    set_opsum_addr(opsum_in_DRAM);

    // GLB address
    set_glb_filter_addr(glb_ifmap_usage);
    set_glb_bias_addr(glb_ifmap_usage + glb_filter_usage);
    set_glb_ofmap_addr(glb_ifmap_usage + glb_filter_usage + glb_bias_usage);

    // activation lengths
    set_input_activation_len(ifmap_len);
    set_output_activation_len(ofmap_len);

    // set_enable(uint32_t scale_factor, bool maxpool, bool relu, bool operation) 
    set_enable(scale, false, true, 0);

    get_dla_hal()->wait_for_irq();
    dla_stop();
#ifdef DLA_INFO
    dump_dla_info_to_csv(DLA_INFO_CSV, "qconv2d_relu", m, e, p, q, r, t, PAD, U,
                         R, S, C, M, W, H);
#endif
    return 0;
};

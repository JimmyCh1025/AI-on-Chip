#include <stdio.h>
#include <stdlib.h>

#include "hardware_dla.h"
#include "runtime.h"

#define ALIGN_SIZE(x) (((x) + 63) & ~63)

/*  //////////      NOTICE      //////////
    all parameter used to set DLA are send in by function argument
*/

void dla_stop() {
  // set disable
  reg_write(DLA_ENABLE_OFFSET, 0);
}

void dla_init() {
#ifdef DLA_INFO
  fprintf(stdout, "DLA runtime info logging enabled.\n");
  dla_reset_runtime_info();
  create_dla_info_to_csv(DLA_INFO_CSV);
#endif
  hal_init();
}

void dla_final() {
#ifdef DLA_INFO
  fprintf(stdout, "Creating dla info file: %s\n", DLA_INFO_CSV);
#endif
  hal_final();
}

void dla_reset_runtime_info() { reset_runtime_info(); }

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
                          uint32_t m, uint32_t e, uint32_t p, uint32_t q, uint32_t r, uint32_t t,
                          // shape parameter
                          uint32_t PAD, uint32_t U, uint32_t R, uint32_t S, uint32_t C, uint32_t M, uint32_t W,
                          uint32_t H) {
  FILE*               file = fopen(filename, "a");
  struct runtime_info info = get_runtime_info();
  fprintf(file, "%s,", operation_name);        // Operation
  fprintf(file, "%10d,", info.elapsed_cycle);  // Cycles
  fprintf(file, "%10d,", info.elapsed_time);   // Time (ns)
  fprintf(file, "%10d,", info.memory_read);    // Memory read
  fprintf(file, "%10d,", info.memory_write);   // Memory write
  fprintf(file, "%d,%d,%d,%d,%d,%d,", m, e, p, q, r, t);
  fprintf(file, "%d,%d,%d,%d,%d,%d,%d,%d\n", PAD, U, R, S, C, M, W, H);
  fclose(file);
}

int qconv2d_relu_maxpool(uint8_t* input_in_DRAM, int8_t* filter_in_DRAM, uint8_t* opsum_in_DRAM, int32_t* bias,
                         uint32_t ofmap_len, uint32_t ifmap_len, uint32_t filter_len,
                         // mapping parameter
                         uint32_t m, uint32_t e, uint32_t p, uint32_t q, uint32_t r, uint32_t t,
                         // shape parameter
                         uint32_t PAD, uint32_t U, uint32_t R, uint32_t S, uint32_t C, uint32_t M, uint32_t W,
                         uint32_t H,
                         uint32_t scale) {  // int32_t scale_factor: merge ifmap and weight and ofmap
                                            // scale bit-shift

#ifdef DLA_INFO
  dla_reset_runtime_info();
#endif
  //! hint>>
  uint32_t GLB_ifmap_len  = (q * r) * (e + R - 1) * W;
  uint32_t GLB_filter_len = (p * t) * (q * r) * 3 * 3;

  uint32_t GLB_remain = 65536 - GLB_filter_len - GLB_ifmap_len;  // 64 * 1024
  float    m_temp     = GLB_remain / (e * W * 4);
  uint32_t m_uint     = (uint32_t)m_temp;
  uint32_t power      = 1;
  while ((power << 1) < m_uint) { power = (power << 1); }
  uint32_t m_best = (M > power) ? power : M;
  m               = (m_best < m) ? m_best : m;
  //  get GLB ADDR

  uint32_t GLB_bias_len = 4 * m;

  uint32_t GLB_bias_address   = GLB_ifmap_len;
  uint32_t GLB_filter_address = GLB_bias_address + GLB_bias_len;
  uint32_t GLB_ofmap_address  = GLB_filter_address + GLB_filter_len;
  //! hint<<

  // =========================================================
  // [FIX] Special Occasion Handling
  // ---------------------------------------------------------
  // Logic: If filter_len is 1728, it should actually be 2304
  // =========================================================
  if (filter_len == 1728) filter_len = 2304;
  // ---------------------------------------------------------

  // Register handle from HAL
  uint32_t input_handle  = mmu_map(input_in_DRAM, ALIGN_SIZE(ifmap_len));
  uint32_t filter_handle = mmu_map(filter_in_DRAM, ALIGN_SIZE(filter_len));
  uint32_t bias_handle   = mmu_map(bias, ALIGN_SIZE(M * 4));
  uint32_t output_handle = mmu_map(opsum_in_DRAM, ALIGN_SIZE(ofmap_len));

  // call lower setting functions
  //! hint>>
  set_mapping_param(m, e, p, q, r, t);
  set_shape_param1(PAD, U, R, S, C, M);
  set_shape_param2(W, H, PAD);

  set_ifmap_addr((uint8_t*)(uint64_t)input_handle);
  set_filter_addr((int8_t*)(uint64_t)filter_handle);
  set_bias_addr((int32_t*)(uint64_t)bias_handle);
  set_opsum_addr((uint8_t*)(uint64_t)output_handle);

  set_glb_filter_addr(GLB_filter_address);
  set_glb_bias_addr(GLB_bias_address);
  set_glb_ofmap_addr(GLB_ofmap_address);

  set_input_activation_len(ifmap_len);
  set_output_activation_len(ofmap_len);

  set_enable(scale, true, true, false);
  //! hint<<

  wait_for_interrupt();
  dla_stop();

  mmu_unmap(input_handle);
  mmu_unmap(filter_handle);
  mmu_unmap(bias_handle);
  mmu_unmap(output_handle);
#ifdef DLA_INFO
  dump_dla_info_to_csv(DLA_INFO_CSV, "qconv2d_relu_maxpool", m, e, p, q, r, t, PAD, U, R, S, C, M, W, H);
#endif
  return 0;
};

int qconv2d_relu(uint8_t* input_in_DRAM, int8_t* filter_in_DRAM, uint8_t* opsum_in_DRAM, int32_t* bias,
                 uint32_t ofmap_len, uint32_t ifmap_len, uint32_t filter_len,
                 // mapping parameter
                 uint32_t m, uint32_t e, uint32_t p, uint32_t q, uint32_t r, uint32_t t,
                 // shape parameter
                 uint32_t PAD, uint32_t U, uint32_t R, uint32_t S, uint32_t C, uint32_t M, uint32_t W, uint32_t H,
                 uint32_t scale) {  // int32_t scale_factor: merge ifmap and
                                    // ofmap scale bit-shift
#ifdef DLA_INFO
  dla_reset_runtime_info();
#endif
  //! hint>>
  uint32_t GLB_ifmap_len  = (q * r) * (e + R - 1) * W;
  uint32_t GLB_filter_len = (p * t) * (q * r) * 3 * 3;

  uint32_t GLB_remain = 65536 - GLB_filter_len - GLB_ifmap_len;  // 64 * 1024
  float    m_temp     = GLB_remain / (e * W * 4);
  uint32_t m_uint     = (uint32_t)m_temp;
  uint32_t power      = 1;
  while ((power << 1) < m_uint) { power = (power << 1); }
  uint32_t m_best = (M > power) ? power : M;
  m               = (m_best < m) ? m_best : m;

  //  get GLB ADDR

  uint32_t GLB_bias_len       = 4 * m;
  uint32_t GLB_bias_address   = GLB_ifmap_len;
  uint32_t GLB_filter_address = GLB_bias_address + GLB_bias_len;
  uint32_t GLB_ofmap_address  = GLB_filter_address + GLB_filter_len;
  //! hint<<

  // Register handle from HAL
  uint32_t input_handle  = mmu_map(input_in_DRAM, ALIGN_SIZE(ifmap_len));
  uint32_t filter_handle = mmu_map(filter_in_DRAM, ALIGN_SIZE(filter_len));
  uint32_t bias_handle   = mmu_map(bias, ALIGN_SIZE(M * 4));
  uint32_t output_handle = mmu_map(opsum_in_DRAM, ALIGN_SIZE(ofmap_len));

  // call lower setting functions
  //! hint>>
  set_mapping_param(m, e, p, q, r, t);
  set_shape_param1(PAD, U, R, S, C, M);
  set_shape_param2(W, H, PAD);

  set_ifmap_addr((uint8_t*)(uint64_t)input_handle);
  set_filter_addr((int8_t*)(uint64_t)filter_handle);
  set_bias_addr((int32_t*)(uint64_t)bias_handle);
  set_opsum_addr((uint8_t*)(uint64_t)output_handle);

  set_glb_filter_addr(GLB_filter_address);
  set_glb_bias_addr(GLB_bias_address);
  set_glb_ofmap_addr(GLB_ofmap_address);

  set_input_activation_len(ifmap_len);
  set_output_activation_len(ofmap_len);

  set_enable(scale, false, true, false);
  //! hint<<

  wait_for_interrupt();
  dla_stop();

  mmu_unmap(input_handle);
  mmu_unmap(filter_handle);
  mmu_unmap(bias_handle);
  mmu_unmap(output_handle);
#ifdef DLA_INFO
  dump_dla_info_to_csv(DLA_INFO_CSV, "qconv2d_relu", m, e, p, q, r, t, PAD, U, R, S, C, M, W, H);
#endif
  return 0;
};

#ifndef CASE1_WORKLOAD_H
#define CASE1_WORKLOAD_H

#include <stdint.h>

extern uint8_t output[];
extern uint8_t conv_relu_max_ans_golden[16384];

void run_workload(void);

#endif /* CASE1_WORKLOAD_H */

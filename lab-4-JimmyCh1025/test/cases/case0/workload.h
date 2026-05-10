#ifndef CASE0_WORKLOAD_H
#define CASE0_WORKLOAD_H

#include <stdint.h>

extern uint8_t output[];
extern uint8_t conv_relu_ans_golden[1024];

void run_workload(void);

#endif /* CASE0_WORKLOAD_H */

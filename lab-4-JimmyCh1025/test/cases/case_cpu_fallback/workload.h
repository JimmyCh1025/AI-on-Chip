#ifndef CASE_CPU_FALLBACK_WORKLOAD_H
#define CASE_CPU_FALLBACK_WORKLOAD_H

#include <stdint.h>

extern uint8_t output[];
extern uint8_t linear_golden[256];
extern uint8_t linear_relu_golden[256];

void run_workload(void);

#endif /* CASE_CPU_FALLBACK_WORKLOAD_H */

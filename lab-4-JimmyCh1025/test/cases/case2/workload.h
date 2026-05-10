#ifndef CASE2_WORKLOAD_H
#define CASE2_WORKLOAD_H

#include <stdint.h>

extern uint8_t input_tensor[];
extern int8_t weight_tensor[];
extern int32_t bias_tensor[];
extern uint8_t output[];

void run_workload(void);

#endif /* CASE2_WORKLOAD_H */

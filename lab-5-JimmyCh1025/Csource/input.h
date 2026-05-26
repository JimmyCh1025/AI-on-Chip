#ifndef INPUT_H
#define INPUT_H
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define ASSERT_MALLOC(ptr)              \
  if ((ptr) == NULL) {                  \
    perror("Memory allocation failed"); \
    exit(EXIT_FAILURE);                 \
  }

struct test_data {
  uint32_t num_classes;
  uint32_t num_per_classes;
  uint32_t data_size;
  char**   classes_name;
  float*** input_data;
};
struct test_data load_bin_data(const char* filename);
void             free_test_data(struct test_data);
#endif

#include "input.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

struct test_data load_bin_data(const char* filename) {
  struct test_data data;

  FILE* file = fopen(filename, "rb");
  if (!file) {
    fprintf(stderr, "Error opening %s\n", filename);
    exit(EXIT_FAILURE);
  }

  fread(&data.num_classes, sizeof(uint32_t), 1, file);

  data.classes_name = (char**)malloc(sizeof(char*) * data.num_classes);
  ASSERT_MALLOC(data.classes_name);

  for (int i = 0; i < data.num_classes; i++) {
    uint32_t name_len;
    fread(&name_len, sizeof(uint32_t), 1, file);
    data.classes_name[i] = (char*)malloc(name_len + 2);
    ASSERT_MALLOC(data.classes_name[i]);
    fread(data.classes_name[i], sizeof(char), name_len, file);
    data.classes_name[i][name_len] = '\0';
  }

  fread(&data.num_per_classes, sizeof(uint32_t), 1, file);
  fread(&data.data_size, sizeof(uint32_t), 1, file);

  data.input_data = (float***)malloc(sizeof(float**) * data.num_classes);
  ASSERT_MALLOC(data.input_data);
  for (int c = 0; c < data.num_classes; c++) {
    data.input_data[c] = (float**)malloc(sizeof(float*) * data.num_per_classes);
    ASSERT_MALLOC(data.input_data[c]);
    for (int n = 0; n < data.num_per_classes; n++) {
      data.input_data[c][n] = (float*)malloc(sizeof(float) * data.data_size);
      ASSERT_MALLOC(data.input_data[c][n]);
      fread(data.input_data[c][n], sizeof(float), data.data_size, file);
    }
  }

  fclose(file);
  return data;
}

void free_test_data(struct test_data data) {
  if (data.classes_name != NULL) {
    for (int i = 0; i < data.num_classes; i++) {
      if (data.classes_name[i] != NULL) { free(data.classes_name[i]); }
    }
    free(data.classes_name);
  }

  if (data.input_data != NULL) {
    for (int c = 0; c < data.num_classes; c++) {
      if (data.input_data[c] != NULL) {
        for (int n = 0; n < data.num_per_classes; n++) {
          if (data.input_data[c][n] != NULL) { free(data.input_data[c][n]); }
        }
        free(data.input_data[c]);
      }
    }
    free(data.input_data);
  }
}

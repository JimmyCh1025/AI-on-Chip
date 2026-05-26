#ifndef UTILS_H
#define UTILS_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define IMAGE_SIZE    1024
#define IMAGE_CHANNEL 3

void normalize(float* input_image, float* output_image);
int  softmax(float* predict, float* output, int size);
int  argmax(float* predict, int size);

struct parsed_arg {
  char* input_file;
  char* weight_file;
  int   class_index;
  int   image_index;
};

int parse_arg(int argc, char** argv, struct parsed_arg* arg);

#endif
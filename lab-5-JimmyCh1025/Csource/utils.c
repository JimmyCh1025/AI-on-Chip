#include "utils.h"

void normalize(float* input_image, float* output_image) {
  float mean[] = {0.4914, 0.4822, 0.4465};
  float std[]  = {0.247, 0.243, 0.261};

  for (int channel = 0; channel < IMAGE_CHANNEL; channel++) {
    for (int pixel = 0; pixel < IMAGE_SIZE; pixel++) {
      output_image[channel * IMAGE_SIZE + pixel] =
          (input_image[channel * IMAGE_SIZE + pixel] - mean[channel]) / std[channel];
    }
  }
}

int softmax(float* predict, float* output, int size) {
  float sum     = 0;
  int   max_arg = 0;
  for (int i = 0; i < size; i++) {
    output[i] = exp(predict[i]);
    sum += output[i];
    if (predict[i] > predict[max_arg]) { max_arg = i; }
  }
  for (int i = 0; i < size; i++) { output[i] /= sum; }
  return max_arg;
}

int argmax(float* predict, int size) {
  int max_arg = 0;
  for (int i = 0; i < size; i++) {
    if (predict[i] > predict[max_arg]) { max_arg = i; }
  }
  return max_arg;
}

int parse_arg(int argc, char** argv, struct parsed_arg* arg) {
  // Check if the arguments are sufficient
  if (argc != 9) {
    fprintf(stdout, "Usage: -i [input.bin] -w [weight.bin] -c [class_index(0)] -n [image_index(0)]\n\n");
    return 1;  // Return with error code
  }

  // Parse arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
      arg->input_file = argv[i + 1];
      i++;  // Skip the next argument because it's the value for -i
    } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
      arg->weight_file = argv[i + 1];
      i++;  // Skip the next argument because it's the value for -w
    } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
      if (sscanf(argv[i + 1], "%d", &arg->class_index) != 1) {
        fprintf(stdout, "Error: Invalid class index value.\n");
        return 1;
      }
      i++;  // Skip the next argument because it's the value for -c
    } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      if (sscanf(argv[i + 1], "%d", &arg->image_index) != 1) {
        fprintf(stdout, "Error: Invalid image index value.\n");
        return 1;
      }
      i++;  // Skip the next argument because it's the value for -n
    } else {
      fprintf(stdout, "Invalid argument: %s\n", argv[i]);
      return 1;
    }
  }

  // Ensure all required arguments are provided
  if (!arg->input_file || !arg->weight_file) {
    fprintf(stdout, "Missing required arguments.\n");
    return 1;
  }
  return 0;
}
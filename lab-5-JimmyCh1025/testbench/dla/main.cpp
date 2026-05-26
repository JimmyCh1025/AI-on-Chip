// testbench/dla/main.cpp
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "color.h"
#include "input.h"
#include "runtime.h"
#include "utils.h"
}

extern "C" int32_t tvmgen_default___tvm_main__(void* input, void* output);

/*
    Usage: -i [(input.bin)] -w [(dummy)] -c [class_index(0)] -n [image_index(0)]
*/
int main(int argc, char* argv[]) {
  setbuf(stdout, NULL);  // Immediate output

  struct test_data  data;
  struct parsed_arg arg = {0};

  if (parse_arg(argc, argv, &arg) != 0 || arg.input_file == NULL) {
    fprintf(stderr, "Error: Invalid arguments or missing input file.\n");
    return 1;
  }

  // Proceed with the rest of the program logic
  fprintf(stdout, "=============================================\n");
  fprintf(stdout, "Input file: %s\n", arg.input_file);
  fprintf(stdout, "Class index: %d\n", arg.class_index);
  fprintf(stdout, "Image index: %d\n", arg.image_index);
  fprintf(stdout, "=============================================\n");

  // load data
  data = load_bin_data(arg.input_file);

  fprintf(stdout, "Image Test: %d/10 image class %12s  \n", arg.image_index, data.classes_name[arg.class_index]);

  // activate dla
  dla_init();

  float model_out[10]          = {0};
  float predict[10]            = {0};
  float normalized_image[3072] = {0};  // 1*3*32*32

  // preprocess
  normalize(data.input_data[arg.class_index][arg.image_index], normalized_image);

  if (tvmgen_default___tvm_main__(normalized_image, model_out) != 0) {
    fprintf(stderr, "TVM model execution failed!\n");
  }

  // softmax layer of predict
  int max_arg = softmax(model_out, predict, 10);

  // print result
  fprintf(stdout, "\n\n=============================================\n");
  for (int i = 0; i < 10; i++) {
    if (arg.class_index == i) {
      fprintf(stdout, GREEN("[%12s]"), data.classes_name[i]);
    } else {
      fprintf(stdout, "[%12s]", data.classes_name[i]);
    }

    if (max_arg == i) {
      if (arg.class_index == i) {
        fprintf(stdout, GREEN("%8.3f%%\n"), predict[i] * 100);
      } else {
        fprintf(stdout, RED("%8.3f%%\n"), predict[i] * 100);
      }
    } else {
      fprintf(stdout, "%8.3f%%\n", predict[i] * 100);
    }
  }
  fprintf(stdout, "=============================================\n");

  // free data
  free_test_data(data);

  // finalize dla
  dla_final();
  return 0;
}
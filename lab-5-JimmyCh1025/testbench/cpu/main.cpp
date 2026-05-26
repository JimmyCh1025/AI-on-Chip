// testbench/cpu/main.cpp
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

void single_test(const struct test_data data, const struct parsed_arg arg);
void full_test(const struct test_data data, const struct parsed_arg arg);

int main(int argc, char* argv[]) {
  setbuf(stdout, NULL);  // Immediate output

  struct test_data  data;
  struct parsed_arg arg = {0};

  if (parse_arg(argc, argv, &arg) != 0 || arg.input_file == NULL) {
    fprintf(stderr, "Error: Invalid arguments or missing input file.\n");
    return 1;
  }

  // load data
  data = load_bin_data(arg.input_file);

  if (arg.class_index != -1 && arg.image_index != -1) {
    single_test(data, arg);
  } else {
    full_test(data, arg);
  }

  // free data
  free_test_data(data);

  return 0;
}

void single_test(const struct test_data data, const struct parsed_arg arg) {
  fprintf(stdout, "===============[ single test ]===============\n");
  fprintf(stdout, "Input file: %s\n", arg.input_file);
  fprintf(stdout, "Class index: %d\n", arg.class_index);
  fprintf(stdout, "Image index: %d\n", arg.image_index);
  fprintf(stdout, "=============================================\n");
  fprintf(stdout, "Image Test: %d/10 image class %12s  \n", arg.image_index, data.classes_name[arg.class_index]);

  float model_out[10]          = {0};
  float predict[10]            = {0};
  float normalized_image[3072] = {0};

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
}

void full_test(const struct test_data data, const struct parsed_arg arg) {
  fprintf(stdout, "================[ full test ]================\n");
  fprintf(stdout, "Input file: %s\n", arg.input_file);
  fprintf(stdout, "=============================================\n");
  fprintf(stdout, GREEN("\'.\'") " is " GREEN("PASS") "," YELLOW("\'<num>\'") " is the " YELLOW("wrong prediction\n"));

  float model_out[10]          = {0};
  float normalized_image[3072] = {0};

  int total = 0, correct = 0;

  fprintf(stdout, "\n\n=============================================\n");
  for (int c = 0; c < data.num_classes; c++) {
    fprintf(stdout, "[%2d - %15s]  ", c, data.classes_name[c]);
    for (int n = 0; n < data.num_per_classes; n++) {
      // preprocess (update normalized_image memory content)
      normalize(data.input_data[c][n], normalized_image);

      // model predict
      if (tvmgen_default___tvm_main__(normalized_image, model_out) != 0) {
        fprintf(stderr, "TVM model execution failed!\n");
      }

      // top-1 instead of softmax layer of prediction
      int max_arg = argmax(model_out, 10);

      // verify result
      if (max_arg == c) {
        fprintf(stdout, GREEN(". "));
        correct++;
      } else {
        fprintf(stdout, YELLOW("%d "), max_arg);
      }
      total++;
    }
    fprintf(stdout, "\n");
  }

  fprintf(stdout, VIBRANT_BLUE("\nCorrect") "/" BRIGHT_BLUE("Total: "));
  fprintf(stdout, VIBRANT_BLUE("%d") "/" BRIGHT_BLUE("%d\n"), correct, total);
  fprintf(stdout, VIBRANT_BLUE("Accuracy: "));
  fprintf(stdout, VIBRANT_BLUE("%.2f%%\n"), total == 0 ? 0.0 : (100.0 * correct / total));

  fprintf(stdout, "=============================================\n");
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int fibonacci(int n) {
  if (n <= 1) { return n; }
  return fibonacci(n - 1) + fibonacci(n - 2);
}

int main() {
  // Allocate memory on the heap
  int* fib_array = (int*)malloc(20 * sizeof(int));  // Allocate space for 20 integers
  if (fib_array == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
  }

  // Calculate Fibonacci numbers and store them in the allocated memory
  for (int i = 0; i < 20; i++) { fib_array[i] = fibonacci(i); }

  // Print the Fibonacci numbers
  printf("Fibonacci sequence:\n");
  for (int i = 0; i < 20; i++) { printf("%d ", fib_array[i]); }
  printf("\n");

  // Release the allocated memory
  free(fib_array);
  printf("Memory released.\n");

  return 0;
}
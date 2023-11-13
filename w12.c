#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef DEBUG
#include <cblas.h>
#endif

typedef struct {
  uint8_t *vector;
  uint8_t *matrix_row;
  size_t size;
  uint64_t result;
} thread_data;

/*** Insert your own matrix & vector data, functions ***/
void *thread_mvm(void *arg) {
  thread_data *data = (thread_data *)arg;
  uint64_t sum = 0;
  for (size_t i = 0; i < data->size; i++) {
    sum += data->matrix_row[i] * data->vector[i];
  }
  data->result = sum;
  return NULL;
}

#ifdef DEBUG
void debug_matrix_multiplication(const uint8_t *matrix, const uint8_t *vector,
                                 const uint64_t *result, uint64_t row_size,
                                 uint64_t col_size) {

  float *matrix_float = malloc(row_size * col_size * sizeof(float));
  float *vector_float = malloc(col_size * sizeof(float));
  float *result_float = malloc(row_size * sizeof(float));

  for (size_t i = 0; i < row_size * col_size; i++) {
    matrix_float[i] = (float)matrix[i];
  }
  for (size_t i = 0; i < col_size; i++) {
    vector_float[i] = (float)vector[i];
  }

  cblas_sgemv(CblasRowMajor, CblasNoTrans, (int32_t)row_size, (int32_t)col_size,
              1.0f, matrix_float, (int32_t)col_size, vector_float, 1, 0.0f,
              result_float, 1);

  // Print result
  printf("*** Answer ***\n");
  for (size_t i = 0; i < row_size; i++) {
    printf("[ %d ]\n", (int)result_float[i]);
  }
  printf("\n");

  // Compare result
  for (size_t i = 0; i < row_size; i++) {
    if (result[i] != (uint64_t)result_float[i]) {
      printf("Error: Wrong answer! (%lu, %lu)\n", result[i],
             (uint64_t)result_float[i]);
      free(matrix_float);
      free(vector_float);
      free(result_float);
      exit(1);
    }
  }
  free(matrix_float);
  free(vector_float);
  free(result_float);
}
#endif

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <row> <column>\n", argv[0]);
    exit(1);
  }

  srand(time(NULL));

  size_t row_size = strtol(argv[1], NULL, 10);
  size_t col_size = strtol(argv[2], NULL, 10);
  pthread_t tid[row_size];
  thread_data t_data[row_size];
  uint8_t matrix[row_size][col_size];
  uint8_t vector[col_size];
  uint64_t result[row_size];

  /*** Insert your code ***/
  // Initialize matrix & vector with random values (multi-threading not needed)

  // Print matrix & vector
  printf("*** Matrix ***\n");
  for (size_t i = 0; i < row_size; i++) {
    for (size_t j = 0; j < col_size; j++) {
      printf("[ %d ]", matrix[i][j]);
      if (j != col_size - 1)
        printf(" ");
    }
    printf("\n\n");
  }

  printf("*** Vector ***\n");
  for (size_t i = 0; i < col_size; i++) {
    printf("[ %d ] ", vector[i]);
    if (i != col_size - 1)
      printf(" ");
  }
  printf("\n\n");

  // Create threads
  // Join threads
  // Store result in `result`
  // Print result
  printf("*** Result ***\n");

  // Print per column
  for (size_t i = 0; i < row_size; i++) {
    printf("[ %lu ]\n", result[i]);
  }
  printf("\n");

#ifdef DEBUG
  debug_matrix_multiplication((uint8_t *)matrix, vector, result, row_size,
                              col_size);
#endif

  return 0;
}

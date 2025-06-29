#include "matrix_ops_pthread.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define WORKLOAD_THRESHOLD 10000

static pthread_t *threads = NULL;
static int pool_size = 0;
__attribute__((constructor)) static void init_threads_pool() {
  pool_size = sysconf(_SC_NPROCESSORS_ONLN);
  threads = malloc(pool_size * sizeof(pthread_t));
  if (!threads) {
    perror("Failed to allocate thread pool");
    exit(EXIT_FAILURE);
  }
}
__attribute__((destructor)) static void cleanup_threads_pool() {
  free(threads);
  threads = NULL;
}

void free_matrix(double **matrix, const int rows) {
  if (matrix == NULL) return;

  for (int row = 0; row < rows; row++) {
    free(matrix[row]);
  }
  free(matrix);
}

double **augment_with_identity(const double *const *target_matrix, const int n_size) {
  double **augmented_matrix = malloc(n_size * sizeof(double *));
  if (!augmented_matrix) return NULL;

  for (int i = 0; i < n_size; i++) {
    augmented_matrix[i] = calloc(2 * n_size, sizeof(double));
    if (!augmented_matrix[i]) {
      free_matrix(augmented_matrix, i);
      return NULL;
    }

    memcpy(augmented_matrix[i], target_matrix[i], n_size * sizeof(double));
    augmented_matrix[i][i + n_size] = 1.0;
  }

  return augmented_matrix;
}

double **slice_invert_from_augmented_matrix(double **augmented_matrix, const int matrix_dimension) {
  double **inverted_matrix = malloc(matrix_dimension * sizeof(double *));
  if (!inverted_matrix) return NULL;

  for (int row = 0; row < matrix_dimension; row++) {
    inverted_matrix[row] = malloc(matrix_dimension * sizeof(double));
    if (!inverted_matrix[row]) {
      free_matrix(inverted_matrix, row);
      return NULL;
    }
    memcpy(inverted_matrix[row], augmented_matrix[row] + matrix_dimension,
           matrix_dimension * sizeof(double));
  }

  return inverted_matrix;
}

void normalize_pivot_row(double **matrix, int row, int matrix_dimension) {
  double pivot = matrix[row][row];
  int augmented_matrix_dimension = 2 * matrix_dimension;
  for (int col = 0; col < augmented_matrix_dimension; col++) {
    matrix[row][col] /= pivot;
  }
}

void perform_row_operations(double **matrix, int current_row, int matrix_dimension, int start_row,
                            int end_row) {
  int augmented_matrix_dimension = 2 * matrix_dimension;
  for (int r = start_row; r < end_row; r++) {
    if (r != current_row) {
      double factor = matrix[r][current_row];
      for (int c = 0; c < augmented_matrix_dimension; c++) {
        matrix[r][c] -= factor * matrix[current_row][c];
      }
    }
  }
}

struct inverse_thread_data {
  double **augmented_matrix;
  int current_row;
  int start_index;
  int end_index;
  int matrix_dimension;
};

void *inverse_thread_fn(void *arg) {
  struct inverse_thread_data *data = (struct inverse_thread_data *)arg;
  perform_row_operations(data->augmented_matrix, data->current_row, data->matrix_dimension,
                         data->start_index, data->end_index);
  return NULL;
}

double **inverse_matrix(const double *const *matrix, const int size) {
  double **augmented_matrix = augment_with_identity(matrix, size);
  if (!augmented_matrix) {
    fprintf(stderr, "Failed to allocate memory while creating augmented matrix.\n");
    return NULL;
  }

  int workload = 2 * size * size;
  if (workload < (WORKLOAD_THRESHOLD * pool_size)) {
    for (int row_idx = 0; row_idx < size; row_idx++) {
      if (augmented_matrix[row_idx][row_idx] == 0) {
        fprintf(stderr, "Matrix is singular, cannot be inverted.\n");
        free_matrix(augmented_matrix, size);
        return NULL;
      }

      normalize_pivot_row(augmented_matrix, row_idx, size);
      perform_row_operations(augmented_matrix, row_idx, size, 0, size);
    }
  } else {
    for (int row_idx = 0; row_idx < size; row_idx++) {
      if (augmented_matrix[row_idx][row_idx] == 0) {
        fprintf(stderr, "Matrix is singular, cannot be inverted.\n");
        free_matrix(augmented_matrix, size);
        return NULL;
      }
      normalize_pivot_row(augmented_matrix, row_idx, size);

      int rows_per_thread = size / pool_size;
      int remaining_rows = size % pool_size;
      int current_row_start = 0;

      struct inverse_thread_data thread_data[pool_size];

      for (int t = 0; t < pool_size; t++) {
        int chunk_size = rows_per_thread + (t < remaining_rows ? 1 : 0);
        thread_data[t] = (struct inverse_thread_data){
            .augmented_matrix = augmented_matrix,
            .current_row = row_idx,
            .start_index = current_row_start,
            .end_index = current_row_start + chunk_size,
            .matrix_dimension = size,
        };
        current_row_start += chunk_size;
        if (pthread_create(&threads[t], NULL, inverse_thread_fn, &thread_data[t]) != 0) {
          fprintf(stderr, "Failed to create thread.\n");
          free_matrix(augmented_matrix, size);
          return NULL;
        }
      }

      for (int t = 0; t < pool_size; t++) {
        if (pthread_join(threads[t], NULL) != 0) {
          fprintf(stderr, "Failed to join thread.\n");
          free_matrix(augmented_matrix, size);
          return NULL;
        }
      }
    }
  }

  double **inverted_matrix = slice_invert_from_augmented_matrix(augmented_matrix, size);
  if (!inverted_matrix) {
    fprintf(stderr, "Failed to allocate memory while slicing inverted matrix.\n");
    free_matrix(augmented_matrix, size);
    return NULL;
  }

  free_matrix(augmented_matrix, size);
  return inverted_matrix;
}

struct transpose_thread_data {
  const double *const *matrix;
  double **transposed;
  int original_cols;
  int start_row;
  int end_row;
};

// Helper function for transposing matrix (single-threaded)
static void
perform_transpose(const double *const *matrix, double **transposed, int rows, int cols) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      transposed[col][row] = matrix[row][col];
    }
  }
}

void *transpose_thread_fn(void *arg) {
  struct transpose_thread_data *data = (struct transpose_thread_data *)arg;

  for (int i = data->start_row; i < data->end_row; i++) {
    for (int j = 0; j < data->original_cols; j++) {
      data->transposed[j][i] = data->matrix[i][j];
    }
  }

  return NULL;
}

double **transpose_matrix(const double *const *matrix, const int rows, const int cols) {
  double **transposed_matrix = malloc(cols * sizeof(double *));
  if (!transposed_matrix) {
    fprintf(stderr, "Failed to allocate memory for transposed matrix.\n");
    return NULL;
  }

  for (int i = 0; i < cols; i++) {
    transposed_matrix[i] = calloc(rows, sizeof(double));
    if (!transposed_matrix[i]) {
      fprintf(stderr, "Failed to allocate memory for transposed matrix.\n");
      free_matrix(transposed_matrix, i);
      return NULL;
    }
  }

  int workload = rows * cols;
  if (workload < (WORKLOAD_THRESHOLD * pool_size)) {
    perform_transpose(matrix, transposed_matrix, rows, cols);
  } else {
    int rows_per_thread = rows / pool_size;
    int remaining_rows = rows % pool_size;
    int current_row_start = 0;

    struct transpose_thread_data thread_data[pool_size];

    for (int t = 0; t < pool_size; t++) {
      int chunk_size = rows_per_thread + (t < remaining_rows ? 1 : 0);
      thread_data[t] = (struct transpose_thread_data){
          .matrix = matrix,
          .transposed = transposed_matrix,
          .original_cols = cols,
          .end_row = current_row_start + chunk_size,
          .start_row = current_row_start,
      };
      current_row_start += chunk_size;

      if (pthread_create(&threads[t], NULL, transpose_thread_fn, &thread_data[t]) != 0) {
        fprintf(stderr, "Failed to create thread.\n");
        free_matrix(transposed_matrix, cols);
        return NULL;
      }
    }

    for (int t = 0; t < pool_size; t++) {
      if (pthread_join(threads[t], NULL) != 0) {
        fprintf(stderr, "Failed to join thread.\n");
        free_matrix(transposed_matrix, cols);
        return NULL;
      }
    }
  }

  return transposed_matrix;
}

struct multiply_thread_data {
  const double *const *matrix_a;
  const double *const *matrix_b;
  double **result_matrix;
  int start_row;
  int end_row;
  int cols_b;
  int cols_a;
};

// Helper function for matrix multiplication (single-threaded)
static void perform_matrix_multiplication(const double *const *matrix_a,
                                          const double *const *matrix_b, double **result_matrix,
                                          const int rows_a, const int cols_a, const int cols_b) {
  const int BLOCK_SIZE = 64;
  for (int i = 0; i < rows_a; i++) {
    for (int jj = 0; jj < cols_b; jj += BLOCK_SIZE) {
      for (int kk = 0; kk < cols_a; kk += BLOCK_SIZE) {
        int j_end = (jj + BLOCK_SIZE) < cols_b ? (jj + BLOCK_SIZE) : cols_b;
        int k_end = (kk + BLOCK_SIZE) < cols_a ? (kk + BLOCK_SIZE) : cols_a;

        for (int col = jj; col < j_end; col++) {
          double sum = result_matrix[i][col];
          for (int row = kk; row < k_end; row++) {
            sum += matrix_a[i][row] * matrix_b[row][col];
          }
          result_matrix[i][col] = sum;
        }
      }
    }
  }
}

// This version ensures CPU L1 or L2 cache utilization
void *mul_thread_fn(void *arg) {
  struct multiply_thread_data *data = (struct multiply_thread_data *)arg;
  const int BLOCK_SIZE = 64;

  for (int i = data->start_row; i < data->end_row; i++) {
    for (int jj = 0; jj < data->cols_b; jj += BLOCK_SIZE) {
      for (int kk = 0; kk < data->cols_a; kk += BLOCK_SIZE) {
        int j_end = (jj + BLOCK_SIZE) < data->cols_b ? (jj + BLOCK_SIZE) : data->cols_b;
        int k_end = (kk + BLOCK_SIZE) < data->cols_a ? (kk + BLOCK_SIZE) : data->cols_a;

        for (int col = jj; col < j_end; col++) {
          double sum = data->result_matrix[i][col];
          for (int row = kk; row < k_end; row++) {
            sum += data->matrix_a[i][row] * data->matrix_b[row][col];
          }
          data->result_matrix[i][col] = sum;
        }
      }
    }
  }
  return NULL;
}

double **multiply_matrices(const double *const *mat_a, const double *const *mat_b, const int rows_a,
                           const int cols_a, const int rows_b, const int cols_b) {
  if (!mat_a || !mat_b) {
    fprintf(stderr, "Invalid matrix provided for multiplication, got NULL.\n");
    return NULL;
  }

  if (cols_a != rows_b) {
    fprintf(
        stderr,
        "Incompatible matrix dimensions for multiplication, got %d x %d and %d x %d, %d != %d\n",
        rows_a, cols_a, rows_b, cols_b, cols_a, rows_b);
    return NULL;
  }

  double **result_matrix = malloc(rows_a * sizeof(double *));
  if (!result_matrix) {
    fprintf(stderr, "Failed to allocate memory for result matrix.\n");
    return NULL;
  }

  for (int i = 0; i < rows_a; i++) {
    result_matrix[i] = calloc(cols_b, sizeof(double));
    if (!result_matrix[i]) {
      fprintf(stderr, "Failed to allocate memory for result matrix.");
      free_matrix(result_matrix, i);
      return NULL;
    }
  }
  const int workload = rows_a * cols_b * cols_a;
  if (workload < (WORKLOAD_THRESHOLD * pool_size)) {
    perform_matrix_multiplication(mat_a, mat_b, result_matrix, rows_a, cols_a, cols_b);
  } else {
    struct multiply_thread_data thread_data[pool_size];
    int rows_per_thread = rows_a / pool_size;
    int remaining_rows = rows_a % pool_size;
    int current_row_start = 0;

    for (int t = 0; t < pool_size; t++) {
      int chunk_size = rows_per_thread + (t < remaining_rows ? 1 : 0);
      thread_data[t] = (struct multiply_thread_data){
          .matrix_a = mat_a,
          .matrix_b = mat_b,
          .result_matrix = result_matrix,
          .start_row = current_row_start,
          .end_row = current_row_start + chunk_size,
          .cols_b = cols_b,
          .cols_a = cols_a,
      };
      current_row_start += chunk_size;

      if (pthread_create(&threads[t], NULL, mul_thread_fn, &thread_data[t]) != 0) {
        fprintf(stderr, "Failed to create thread.\n");
        free_matrix(result_matrix, rows_a);
        return NULL;
      }
    }

    for (int t = 0; t < pool_size; t++) {
      if (pthread_join(threads[t], NULL) != 0) {
        fprintf(stderr, "Failed to join thread.\n");
        free_matrix(result_matrix, rows_a);
        return NULL;
      }
    }
  }
  return result_matrix;
}

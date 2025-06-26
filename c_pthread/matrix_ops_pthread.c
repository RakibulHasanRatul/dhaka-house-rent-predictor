#include "matrix_ops_pthread.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static pthread_t *threads = NULL;
static int pool_size = 0;

__attribute__((constructor)) static void init_thread_pool() {
  pool_size = sysconf(_SC_NPROCESSORS_ONLN);
  threads = malloc(pool_size * sizeof(pthread_t));
  if (!threads) {
    perror("Failed to allocate thread pool");
    exit(EXIT_FAILURE);
  }
}

__attribute__((destructor)) static void cleanup_thread_pool() {
  free(threads);
  threads = NULL;
}

void free_matrix(double **matrix, const int rows) {
  if (matrix == NULL) return;

  for (int i = 0; i < rows; i++) {
    free(matrix[i]);
  }
  free(matrix);
}

double **construct_identity_matrix(const int n_size) {
  double **i_mat = malloc(n_size * sizeof(double *));
  if (!i_mat) return NULL;

  for (int i = 0; i < n_size; i++) {
    i_mat[i] = calloc(n_size, sizeof(double));
    if (!i_mat[i]) {
      free_matrix(i_mat, i);
      return NULL;
    }
    i_mat[i][i] = 1.0;
  }
  return i_mat;
}

double **augment_with_identity(const double *const *target_matrix, const int n_size) {
  double **aug_mat = malloc(n_size * sizeof(double *));
  if (!aug_mat) return NULL;

  double **identity_mat = construct_identity_matrix(n_size);
  if (!identity_mat) {
    free(aug_mat);
    fprintf(stderr, "Error while allocating memory for identity matrix");
    return NULL;
  }

  for (int i = 0; i < n_size; i++) {
    aug_mat[i] = malloc(2 * n_size * sizeof(double));
    if (!aug_mat[i]) {
      free_matrix(aug_mat, i);
      free_matrix(identity_mat, n_size);
      return NULL;
    }

    memcpy(aug_mat[i], target_matrix[i], n_size * sizeof(double));
    memcpy(aug_mat[i] + n_size, identity_mat[i], n_size * sizeof(double));
  }

  free_matrix(identity_mat, n_size);
  return aug_mat;
}

double **slice_invert_from_augmented_matrix(double **matrix, const int n_size) {
  double **invert = malloc(n_size * sizeof(double *));
  if (!invert) return NULL;

  for (int row = 0; row < n_size; row++) {
    invert[row] = malloc(n_size * sizeof(double));
    if (!invert[row]) {
      free_matrix(invert, row);
      return NULL;
    }
    memcpy(invert[row], matrix[row] + n_size, n_size * sizeof(double));
  }

  return invert;
}

struct mat_inverse_thread_data {
  double **augmented_matrix;
  int row;
  int start;
  int end;
  int n_size;
};

void *inverse_th_fn(void *arg) {
  struct mat_inverse_thread_data *data = (struct mat_inverse_thread_data *)arg;
  double **aug = data->augmented_matrix;
  int row = data->row;
  int n_size = data->n_size;

  for (int i = data->start; i < data->end; i++) {
    if (i != row) {
      double factor = aug[i][row];
      for (int c = 0; c < 2 * n_size; c++) {
        aug[i][c] -= factor * aug[row][c];
      }
    }
  }

  return NULL;
}

double **inverse_matrix(const double *const *matrix, const int n_size) {
  double **augmented_matrix = augment_with_identity(matrix, n_size);
  if (!augmented_matrix) {
    fprintf(stderr, "Failed to allocate memory while creating augmented matrix.\n");
    return NULL;
  }

  int workload = 2 * n_size * n_size;
  if (workload < (10000 * pool_size)) {
    for (int row = 0; row < n_size; row++) {
      double pivot = augmented_matrix[row][row];
      if (pivot == 0) {
        fprintf(stderr, "Matrix is singular, cannot be inverted.\n");
        free_matrix(augmented_matrix, n_size);
        return NULL;
      }

      for (int col = 0; col < 2 * n_size; col++) {
        augmented_matrix[row][col] /= pivot;
      }

      for (int i = 0; i < n_size; i++) {
        if (i != row) {
          double factor = augmented_matrix[i][row];
          for (int c = 0; c < 2 * n_size; c++) {
            augmented_matrix[i][c] -= factor * augmented_matrix[row][c];
          }
        }
      }
    }
  } else {
    int split_per_iter = n_size / pool_size;
    int remaining_iter = n_size % pool_size;
    int current = 0;

    struct mat_inverse_thread_data thread_data[pool_size];

    for (int row = 0; row < n_size; row++) {
      double pivot = augmented_matrix[row][row];
      if (pivot == 0) {
        fprintf(stderr, "Matrix is singular, cannot be inverted.\n");
        free_matrix(augmented_matrix, n_size);
        return NULL;
      }

      for (int col = 0; col < 2 * n_size; col++) {
        augmented_matrix[row][col] /= pivot;
      }

      for (int t = 0; t < pool_size; t++) {
        int chunk = split_per_iter + (t < remaining_iter ? 1 : 0);
        thread_data[t] = (struct mat_inverse_thread_data){
            .augmented_matrix = augmented_matrix,
            .row = row,
            .start = current,
            .end = current + chunk,
            .n_size = n_size,
        };
        current += chunk;
        if (pthread_create(&threads[t], NULL, inverse_th_fn, &thread_data[t]) != 0) {
          fprintf(stderr, "Failed to create thread.\n");
          free_matrix(augmented_matrix, n_size);
          return NULL;
        }
      }

      for (int t = 0; t < pool_size; t++) {
        if (pthread_join(threads[t], NULL) != 0) {
          fprintf(stderr, "Failed to join thread.\n");
          free_matrix(augmented_matrix, n_size);
          return NULL;
        }
      }
    }
  }

  double **inverted_matrix = slice_invert_from_augmented_matrix(augmented_matrix, n_size);
  if (!inverted_matrix) {
    fprintf(stderr, "Failed to allocate memory while slicing inverted matrix.\n");
    free_matrix(augmented_matrix, n_size);
    return NULL;
  }

  free_matrix(augmented_matrix, n_size);
  return inverted_matrix;
}

struct transpose_th_d {
  const double *const *matrix;
  double **transposed;
  int iter_limit;
  int start;
  int end;
};

void *transpose_th_fn(void *arg) {
  struct transpose_th_d *data = (struct transpose_th_d *)arg;

  for (int i = data->start; i < data->end; i++) {
    for (int j = 0; j < data->iter_limit; j++) {
      data->transposed[j][i] = data->matrix[i][j];
    }
  }

  return NULL;
}

double **transpose_matrix(const double *const *matrix, const int rows, const int cols) {
  double **transposed = malloc(cols * sizeof(double *));
  if (!transposed) {
    fprintf(stderr, "Failed to allocate memory for transposed matrix.\n");
    return NULL;
  }

  for (int i = 0; i < cols; i++) {
    transposed[i] = calloc(rows, sizeof(double));
    if (!transposed[i]) {
      fprintf(stderr, "Failed to allocate memory for transposed matrix.\n");
      free_matrix(transposed, i);
      return NULL;
    }
  }

  int workload = rows * cols;
  if (workload < (10000 * pool_size)) {
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        transposed[col][row] = matrix[row][col];
      }
    }

  } else {
    int iter_per_thread = rows / pool_size;
    int remaining_iter = rows % pool_size;
    int current_unit = 0;

    struct transpose_th_d thread_data[pool_size];

    for (int t = 0; t < pool_size; t++) {
      thread_data[t] = (struct transpose_th_d){
          .matrix = matrix,
          .transposed = transposed,
          .iter_limit = cols,
          .end = current_unit + iter_per_thread + (t < remaining_iter ? 1 : 0),
          .start = current_unit,
      };
      current_unit += iter_per_thread;

      if (pthread_create(&threads[t], NULL, transpose_th_fn, &thread_data[t]) != 0) {
        fprintf(stderr, "Failed to create thread.\n");
        return NULL;
      }
    }

    for (int t = 0; t < pool_size; t++) {
      if (pthread_join(threads[t], NULL) != 0) {
        fprintf(stderr, "Failed to join thread.\n");
        return NULL;
      }
    }
  }

  return transposed;
}

struct mul_th_d {
  const double *const *mat_a;
  const double *const *mat_b;
  double **result;
  int start;
  int end;
  int upper_iter;
  int lower_iter;
};

// This version ensures CPU L1 or L2 cache utilization
void *mul_th_fn(void *arg) {
  struct mul_th_d *data = (struct mul_th_d *)arg;
  const int BLOCK_SIZE = 64;

  for (int i = data->start; i < data->end; i++) {
    for (int jj = 0; jj < data->upper_iter; jj += BLOCK_SIZE) {
      for (int kk = 0; kk < data->lower_iter; kk += BLOCK_SIZE) {
        int j_end = (jj + BLOCK_SIZE) < data->upper_iter ? (jj + BLOCK_SIZE) : data->upper_iter;
        int k_end = (kk + BLOCK_SIZE) < data->lower_iter ? (kk + BLOCK_SIZE) : data->lower_iter;

        for (int col = jj; col < j_end; col++) {
          double sum = data->result[i][col];
          for (int row = kk; row < k_end; row++) {
            sum += data->mat_a[i][row] * data->mat_b[row][col];
          }
          data->result[i][col] = sum;
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

  double **result = malloc(rows_a * sizeof(double *));
  if (!result) {
    fprintf(stderr, "Failed to allocate memory for result matrix.\n");
    return NULL;
  }

  for (int i = 0; i < rows_a; i++) {
    result[i] = calloc(cols_b, sizeof(double));
    if (!result[i]) {
      fprintf(stderr, "Failed to allocate memory for result matrix.");
      free_matrix(result, i);
      return NULL;
    }
  }
  const int workload = rows_a * cols_b * cols_a;
  if (workload < (10000 * pool_size)) {
    // also ensure L1 and L2 cache utilization
    const int BLOCK_SIZE = 64;
    for (int i = 0; i < rows_a; i++) {
      for (int jj = 0; jj < cols_b; jj += BLOCK_SIZE) {
        for (int kk = 0; kk < cols_a; kk += BLOCK_SIZE) {
          int j_end = (jj + BLOCK_SIZE) < cols_b ? (jj + BLOCK_SIZE) : cols_b;
          int k_end = (kk + BLOCK_SIZE) < cols_a ? (kk + BLOCK_SIZE) : cols_a;

          for (int col = jj; col < j_end; col++) {
            double sum = result[i][col];
            for (int row = kk; row < k_end; row++) {
              sum += mat_a[i][row] * mat_b[row][col];
            }
            result[i][col] = sum;
          }
        }
      }
    }

  } else {
    struct mul_th_d thread_data[pool_size];
    int iter_per_thread = rows_a / pool_size;
    int remaining = rows_a % pool_size;
    int current = 0;

    for (int t = 0; t < pool_size; t++) {
      int chunk = iter_per_thread + (t < remaining ? 1 : 0);
      thread_data[t] = (struct mul_th_d){
          .mat_a = mat_a,
          .mat_b = mat_b,
          .result = result,
          .start = current,
          .end = current + chunk,
          .upper_iter = cols_b,
          .lower_iter = cols_a,
      };
      current += chunk;

      if (pthread_create(&threads[t], NULL, mul_th_fn, &thread_data[t]) != 0) {
        fprintf(stderr, "Failed to create thread.\n");
        free_matrix(result, rows_a);
        return NULL;
      }
    }

    for (int t = 0; t < pool_size; t++) {
      if (pthread_join(threads[t], NULL) != 0) {
        fprintf(stderr, "Failed to join thread.\n");
        free_matrix(result, rows_a);
        return NULL;
      }
    }
  }
  return result;
}

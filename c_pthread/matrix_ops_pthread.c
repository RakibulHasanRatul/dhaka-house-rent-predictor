#include "matrix_ops_pthread.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// overwhelmed to see that i've wrote memory leak prevention code more than actual logic!

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

void *eliminate_row_range_thread_func(void *arg) {
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

  const long int NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
  int workload = 2 * n_size * n_size;
  if (workload < (10000 * NUM_THREADS)) {
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
    long int total_split = n_size < NUM_THREADS ? n_size : NUM_THREADS;
    int split_per_iter = n_size / total_split;
    int remaining_iter = n_size % total_split;
    int current = 0;

    pthread_t threads[total_split];
    struct mat_inverse_thread_data thread_data[total_split];

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

      for (int t = 0; t < total_split; t++) {
        int chunk = split_per_iter + (t < remaining_iter ? 1 : 0);
        thread_data[t] = (struct mat_inverse_thread_data){
            .augmented_matrix = augmented_matrix,
            .row = row,
            .start = current,
            .end = current + chunk,
            .n_size = n_size,
        };
        current += chunk;
        pthread_create(&threads[t], NULL, eliminate_row_range_thread_func, &thread_data[t]);
      }

      for (int t = 0; t < total_split; t++) {
        pthread_join(threads[t], NULL);
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

struct mat_transpose_thread_data {
  const double *const *matrix;
  double **transposed;
  int iter_max;
  int start;
  int end;
};

void *mat_transpose_thread_func(void *arg) {
  struct mat_transpose_thread_data *data = (struct mat_transpose_thread_data *)arg;

  for (int i = data->start; i < data->end; i++) {
    for (int j = 0; j < data->iter_max; j++) {
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

  const long int NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
  int workload = rows * cols;
  if (workload < (20000 * NUM_THREADS)) {
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        transposed[col][row] = matrix[row][col];
      }
    }

  } else {
    long int total_split = rows < NUM_THREADS ? rows : NUM_THREADS;
    int iter_per_thread = rows / total_split;
    int remaining_iter = rows % total_split;
    int current_unit = 0;

    struct mat_transpose_thread_data thread_data[total_split];
    pthread_t threads[total_split];

    for (int t = 0; t < total_split; t++) {
      thread_data[t] = (struct mat_transpose_thread_data){
          .matrix = matrix,
          .transposed = transposed,
          .iter_max = cols,
          .end = current_unit + iter_per_thread + (t < remaining_iter ? 1 : 0),
          .start = current_unit,
      };
      current_unit += iter_per_thread;

      if (pthread_create(&threads[t], NULL, mat_transpose_thread_func, &thread_data[t]) != 0) {
        fprintf(stderr, "Failed to create thread.\n");
        return NULL;
      }
    }

    for (int t = 0; t < total_split; t++) {
      if (pthread_join(threads[t], NULL) != 0) {
        fprintf(stderr, "Failed to join thread.\n");
        return NULL;
      }
    }
  }

  return transposed;
}

struct mat_mul_thread_data {
  const double *const *mat_a;
  const double *const *mat_b;
  double **result;
  int start;
  int end;
  int upper_iter;
  int lower_iter;
};

void *mat_mul_thread_func(void *arg) {
  struct mat_mul_thread_data *data = (struct mat_mul_thread_data *)arg;

  // Row-wise parallelization: Each thread computes a range of rows
  for (int i = data->start; i < data->end; i++) {
    for (int j = 0; j < data->upper_iter; j++) {
      double sum = 0.0;
      for (int k = 0; k < data->lower_iter; k++) {
        sum += data->mat_a[i][k] * data->mat_b[k][j];
      }
      data->result[i][j] = sum;
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
  const long int NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
  int workload = rows_a * cols_b * cols_a;
  if (workload < (50000 * NUM_THREADS)) {
    for (int i = 0; i < rows_a; i++) {
      for (int j = 0; j < cols_b; j++) {
        for (int k = 0; k < cols_a; k++) {
          result[i][j] += mat_a[i][k] * mat_b[k][j];
        }
      }
    }

  } else {
    int total_split = rows_a < NUM_THREADS ? rows_a : NUM_THREADS;
    pthread_t threads[total_split];
    struct mat_mul_thread_data thread_data[total_split];
    int iter_per_thread = rows_a / total_split;
    int remaining = rows_a % total_split;
    int current = 0;

    for (int t = 0; t < total_split; t++) {
      int chunk = iter_per_thread + (t < remaining ? 1 : 0);
      thread_data[t] = (struct mat_mul_thread_data){
          .mat_a = mat_a,
          .mat_b = mat_b,
          .result = result,
          .start = current,
          .end = current + chunk,
          .upper_iter = cols_b,
          .lower_iter = cols_a,
      };
      current += chunk;

      if (pthread_create(&threads[t], NULL, mat_mul_thread_func, &thread_data[t]) != 0) {
        fprintf(stderr, "Failed to create thread.\n");
        free_matrix(result, rows_a);
        return NULL;
      }
    }

    for (int t = 0; t < total_split; t++) {
      if (pthread_join(threads[t], NULL) != 0) {
        fprintf(stderr, "Failed to join thread.\n");
        free_matrix(result, rows_a);
        return NULL;
      }
    }
  }
  return result;
}

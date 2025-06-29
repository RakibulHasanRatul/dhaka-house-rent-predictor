#include "matrix_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

double **inverse_matrix(const double *const *matrix, const int n_size) {
  double **augmented_matrix = augment_with_identity(matrix, n_size);
  if (!augmented_matrix) {
    fprintf(stderr, "Failed to allocate memory while creating augmented matrix.\n");
    return NULL;
  }

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

  double **inverted_matrix = slice_invert_from_augmented_matrix(augmented_matrix, n_size);
  if (!inverted_matrix) {
    fprintf(stderr, "Failed to allocate memory while slicing inverted matrix.\n");
    free_matrix(augmented_matrix, n_size);
    return NULL;
  }

  free_matrix(augmented_matrix, n_size);
  return inverted_matrix;
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

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      transposed[c][r] = matrix[r][c];
    }
  }

  return transposed;
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

  for (int i = 0; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j++) {
      for (int k = 0; k < cols_a; k++) {
        result[i][j] += mat_a[i][k] * mat_b[k][j];
      }
    }
  }

  return result;
}
#ifndef MATRIX_OPS_OMP_H
#define MATRIX_OPS_OMP_H
void free_matrix(double **matrix, const int rows);
double **inverse_matrix(const double *const *matrix, const int size);
double **multiply_matrices(const double *const *mat_a, const double *const *mat_b, const int rows_a,
                           const int cols_a, const int rows_b, const int cols_b);

double **transpose_matrix(const double *const *matrix, const int rows, const int cols);
#endif

#include "linear_regression_model.h"

#include <stdio.h>

#include "matrix_ops.h"

double **train(const double *const *feature_vectors, const double *const *label_vectors,
               const int n_samples, const int n_vector_elements, const double lambda) {
  // x = features, dimension n_samples x n_features
  double **xt = transpose_matrix(feature_vectors, n_samples, n_vector_elements);
  if (!xt) {
    fprintf(stderr, "Failed to transpose feature_vectors.\n");
    return NULL;
  }
  // xt dimensions = n_features x n_samples
  double **xtx = multiply_matrices((const double *const *)xt, feature_vectors, n_vector_elements,
                                   n_samples, n_samples, n_vector_elements);
  if (!xtx) {
    fprintf(stderr, "Failed to multiply feature_vectors with its transpose.");
    free_matrix(xt, n_vector_elements);
    return NULL;
  }
  // xtx dimensions = n_features x n_features

  // regularize
  for (int i = 0; i < n_vector_elements; i++) {
    xtx[i][i] += lambda;
  }

  double **xtx_inv = inverse_matrix((const double *const *)xtx, n_vector_elements);
  if (!xtx_inv) {
    fprintf(stderr, "Failed to inverse xtx.");
    free_matrix(xt, n_vector_elements);
    free_matrix(xtx, n_vector_elements);
    return NULL;
  }
  // xtx_inv dimensions = n_features x n_features
  double **xtx_inv_xt =
      multiply_matrices((const double *const *)xtx_inv, (const double *const *)xt,
                        n_vector_elements, n_vector_elements, n_vector_elements, n_samples);
  if (!xtx_inv_xt) {
    fprintf(stderr, "Failed to multiply xtx_inv with xt.");
    free_matrix(xt, n_vector_elements);
    free_matrix(xtx, n_vector_elements);
    free_matrix(xtx_inv, n_vector_elements);
    return NULL;
  }
  // xtx_inv_xt dimensions = n_features x n_samples

  double **weights =
      multiply_matrices((const double *const *)xtx_inv_xt, (const double *const *)label_vectors,
                        n_vector_elements, n_samples, n_samples, 1);
  if (!weights) {
    fprintf(stderr, "Failed to calculate weights.\n");
    free_matrix(xt, n_vector_elements);
    free_matrix(xtx, n_vector_elements);
    free_matrix(xtx_inv, n_vector_elements);
    free_matrix(xtx_inv_xt, n_vector_elements);
    return NULL;
  }

  free_matrix(xt, n_vector_elements);
  free_matrix(xtx, n_vector_elements);
  free_matrix(xtx_inv, n_vector_elements);
  free_matrix(xtx_inv_xt, n_vector_elements);

  return weights;
}

double predict(const double *const features, const double *const *weights, const int n_features) {
  double prediction = weights[0][0];
  for (int i = 0; i < n_features; i++) prediction += features[i] * weights[i + 1][0];
  // Why?
  // just to reduce number of mathematical operations,
  // instead of transposing, we can use weights[row][0]
  return prediction;
}
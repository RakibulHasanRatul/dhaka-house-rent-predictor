#ifndef LINEAR_REGRESSION_MODEL_H
#define LINEAR_REGRESSION_MODEL_H

double **train(const double *const *feature_vectors, const double *const *label_vectors,
               const int n_samples, const int n_vector_elements, const double lambda);
double predict(const double *const features, const double *const *weights, const int n_features);

#endif
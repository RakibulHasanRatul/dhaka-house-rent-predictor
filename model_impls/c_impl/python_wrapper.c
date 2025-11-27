#include <python3.13/Python.h>
#include <stdlib.h>

#include "linear_regression_model.h"
#include "matrix_ops.h"

static PyObject* train_function_wrapper(PyObject* self, PyObject* args, PyObject* kwargs) {
  PyObject *feature_vectors_obj, *label_vectors_obj;
  double lambda = 1e-12;

  static char* kwlist[] = {"feature_vectors", "label_vectors", "lambda_", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!|d", kwlist, &PyList_Type,
                                   &feature_vectors_obj, &PyList_Type, &label_vectors_obj,
                                   &lambda)) {
    return NULL;
  }

  int n_samples = PyList_Size(feature_vectors_obj);
  if (n_samples == 0) {
    PyErr_SetString(PyExc_ValueError, "Feature vectors list is empty, got []");
    return NULL;
  }
  int n_features = PyList_Size(PyList_GetItem(feature_vectors_obj, 0));
  if (n_features == 0) {
    PyErr_SetString(PyExc_ValueError, "Feature vectors list is empty, got [[]]");
    return NULL;
  }

  double** feature_vectors = malloc(n_samples * sizeof(double*));
  // parameter to pass in the function
  double** label_vectors = malloc(n_samples * sizeof(double*));

  if (!feature_vectors || !label_vectors) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for feature or label vectors.");
    goto handle_memory_error;
  }

  for (int row = 0; row < n_samples; row++) {
    PyObject* feature_vector_obj = PyList_GetItem(feature_vectors_obj, row);
    PyObject* label_vector_obj = PyList_GetItem(label_vectors_obj, row);

    if (!PyList_Check(feature_vector_obj) || !PyList_Check(label_vector_obj)) {
      PyErr_SetString(PyExc_TypeError, "Expected lists of lists of floats.");
      goto handle_memory_error;
    }

    if (PyList_Size(feature_vector_obj) != n_features) {
      PyErr_SetString(PyExc_ValueError, "Incompatible sizes of feature vector found.");
      goto handle_memory_error;
    }
    if (PyList_Size(label_vector_obj) != 1) {
      PyErr_SetString(PyExc_ValueError, "Incompatible sizes of label vector found.");
      goto handle_memory_error;
    }

    feature_vectors[row] = malloc(n_features * sizeof(double));
    label_vectors[row] = malloc(sizeof(double));

    if (!feature_vectors[row] || !label_vectors[row]) {
      char* message = "Failed to allocate memory for feature or label vectors elements.";
      PyErr_SetString(PyExc_MemoryError, message);
      // isn't necessary, but to look the code a bit cleaner to read
      goto handle_memory_error;
    }

    for (int col = 0; col < n_features; col++) {
      PyObject* feature_obj = PyList_GetItem(feature_vector_obj, col);
      feature_vectors[row][col] = PyFloat_AsDouble(feature_obj);
      if (PyErr_Occurred()) goto handle_memory_error;
    }

    PyObject* label_obj = PyList_GetItem(label_vector_obj, 0);
    // because label vector is shaped as (n_samples x 1)
    label_vectors[row][0] = PyFloat_AsDouble(label_obj);
    if (PyErr_Occurred()) goto handle_memory_error;
  }

  double** weights = train((const double* const*)feature_vectors,
                           (const double* const*)label_vectors, n_samples, n_features, lambda);

  if (!weights) {
    PyErr_SetString(PyExc_RuntimeError, "Training failed.");
    goto handle_memory_error;
  }

  // form python lists
  PyObject* weight_vectors_list = PyList_New(n_features);
  for (int col = 0; col < n_features; col++) {
    PyObject* weight_vector_list = PyList_New(1);
    PyList_SetItem(weight_vector_list, 0, PyFloat_FromDouble(weights[col][0]));
    PyList_SetItem(weight_vectors_list, col, weight_vector_list);
  }

  free_matrix(weights, n_features);
  free_matrix(feature_vectors, n_samples);
  free_matrix(label_vectors, n_samples);

  return weight_vectors_list;

handle_memory_error:
  if (feature_vectors) free_matrix(feature_vectors, n_samples);
  if (label_vectors) free_matrix(label_vectors, n_samples);

  return NULL;
}

static PyObject* predict_function_wrapper(PyObject* self, PyObject* args, PyObject* kwargs) {
  PyObject *features_obj, *weights_obj;

  static char* kwlist[] = {"features", "weights", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!", kwlist, &PyList_Type, &features_obj,
                                   &PyList_Type, &weights_obj)) {
    return NULL;
  }

  int n_features = PyList_Size(features_obj);

  if (n_features == 0) {
    PyErr_SetString(PyExc_ValueError, "Features list is empty, got []");
    return NULL;
  }

  int n_weights = PyList_Size(weights_obj);

  if (n_weights == 0) {
    PyErr_SetString(PyExc_ValueError, "Weights list is empty, got []");
    return NULL;
  }

  if (n_features != (n_weights - 1)) {
    PyErr_SetString(PyExc_ValueError,
                    "Weights should have n+1 x 1 shape where feature should have 1 x n shape .");
    return NULL;
  }

  double* features = malloc(n_features * sizeof(double));
  double** weights = malloc((n_features + 1) * sizeof(double*));

  if (!features || !weights) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for features or weights.");
    goto handle_memory_error;
  }

  for (int col = 0; col < n_features; col++) {
    PyObject* feature_obj = PyList_GetItem(features_obj, col);
    features[col] = PyFloat_AsDouble(feature_obj);
    if (PyErr_Occurred()) goto handle_memory_error;
  }

  for (int row = 0; row < n_weights; row++) {
    weights[row] = malloc(sizeof(double));
    if (!weights[row]) {
      PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for weights elements.");
      goto handle_memory_error;
    }
    PyObject* weight_vector_obj = PyList_GetItem(weights_obj, row);
    PyObject* weight_obj = PyList_GetItem(weight_vector_obj, 0);
    weights[row][0] = PyFloat_AsDouble(weight_obj);
    if (PyErr_Occurred()) goto handle_memory_error;
  }

  double prediction = predict((const double*)features, (const double* const*)weights, n_features);

  // Free allocated memory
  free(features);
  free_matrix(weights, n_features);
  return PyFloat_FromDouble(prediction);

handle_memory_error:
  if (features) free(features);
  if (weights) free_matrix(weights, n_features);
  return NULL;
}

static PyMethodDef methods[] = {
    {"train", (PyCFunction)train_function_wrapper, METH_VARARGS | METH_KEYWORDS,
     "Trains a linear regression model."},
    {"predict", (PyCFunction)predict_function_wrapper, METH_VARARGS | METH_KEYWORDS,
     "Predicts the output using a linear regression model."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "c_impl",
                                    "C implementation of linear regression model", -1, methods};

PyMODINIT_FUNC PyInit_c_impl(void) { return PyModule_Create(&module); }

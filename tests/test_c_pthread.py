import pytest

from c_pthread import train as train_c_impl, predict as predict_c_impl
from app.model.linear_regression import model_train as train_python_impl

# the python implementation is tested for expected results, that is why used here.


def test_model_train_2_features():
    feature_matrix = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    labels = [[11.0], [12.0], [13.0], [14.0], [15.0]]
    expected_result = train_python_impl(feature_matrix, labels)
    result = train_c_impl(feature_matrix, labels)
    assert len(result) == len(expected_result)
    assert len(result[0]) == len(expected_result[0])
    for i in range(len(result)):
        for j in range(len(result[0])):
            assert abs(result[i][j] - expected_result[i][j]) < 1e-3


def test_model_train_with_regularization_low_lambda():
    feature_matrix = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    labels = [[11.0], [12.0], [13.0], [14.0], [15.0]]
    lambda_l2 = 0.001
    expected_result = train_python_impl(feature_matrix, labels, lambda_l2)
    result = train_c_impl(feature_matrix, labels, lambda_l2)
    assert len(result) == len(expected_result)
    assert len(result[0]) == len(expected_result[0])
    for i in range(len(result)):
        for j in range(len(result[0])):
            assert abs(result[i][j] - expected_result[i][j]) < 1e-3


def test_model_train_with_regularization_high_lambda():
    feature_matrix = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    labels = [[11.0], [12.0], [13.0], [14.0], [15.0]]
    lambda_l2 = 100.0
    expected_result = train_python_impl(feature_matrix, labels, lambda_l2)
    result = train_c_impl(feature_matrix, labels, lambda_l2)
    assert len(result) == len(expected_result)
    assert len(result[0]) == len(expected_result[0])
    for i in range(len(result)):
        for j in range(len(result[0])):
            assert abs(result[i][j] - expected_result[i][j]) < 1e-3


def test_model_train_with_negative_values():
    feature_matrix = [[-1.0, 2.0], [3.0, -4.0], [5.0, 6.0], [7.0, -8.0], [9.0, -10.0]]
    labels = [[11.0], [-12.0], [13.0], [-14.0], [15.0]]
    expected_result = train_python_impl(feature_matrix, labels)
    result = train_c_impl(feature_matrix, labels)
    assert len(result) == len(expected_result)
    assert len(result[0]) == len(expected_result[0])
    for i in range(len(result)):
        for j in range(len(result[0])):
            assert abs(result[i][j] - expected_result[i][j]) < 1e-3


def test_model_train_with_negative_values_and_regularization():
    feature_matrix = [[-1.0, 2.0], [3.0, -4.0], [5.0, 6.0], [7.0, -8.0], [9.0, -10.0]]
    labels = [[11.0], [-12.0], [13.0], [-14.0], [15.0]]
    lambda_l2 = 0.01
    expected_result = train_python_impl(feature_matrix, labels, lambda_l2)
    result = train_c_impl(feature_matrix, labels, lambda_l2)
    assert len(result) == len(expected_result)
    assert len(result[0]) == len(expected_result[0])
    for i in range(len(result)):
        for j in range(len(result[0])):
            assert abs(result[i][j] - expected_result[i][j]) < 1e-3


def test_model_train_with_negative_values_and_high_regularization():
    feature_matrix = [[-1.0, 2.0], [3.0, -4.0], [5.0, 6.0], [7.0, -8.0], [9.0, -10.0]]
    labels = [[11.0], [-12.0], [13.0], [-14.0], [15.0]]
    lambda_l2 = 100.0
    expected_result = train_python_impl(feature_matrix, labels, lambda_l2)
    result = train_c_impl(feature_matrix, labels, lambda_l2)
    assert len(result) == len(expected_result)
    assert len(result[0]) == len(expected_result[0])
    for i in range(len(result)):
        for j in range(len(result[0])):
            assert abs(result[i][j] - expected_result[i][j]) < 1e-3


def test_model_train_no_samples():
    feature_matrix: list[list[float]] = []
    labels: list[list[float]] = []
    with pytest.raises(Exception):
        train_c_impl(feature_matrix, labels)


def test_model_train_no_features():
    feature_matrix: list[list[float]] = [[]]
    labels: list[list[float]] = [[1.0]]
    with pytest.raises(Exception):
        train_c_impl(feature_matrix, labels)


def test_model_train_one_sample():
    feature_matrix = [[1.0, 2.0]]
    labels = [[1.0]]
    expected_result = train_python_impl(feature_matrix, labels)
    result = train_c_impl(feature_matrix, labels)
    assert len(result) == len(expected_result)
    assert len(result[0]) == len(expected_result[0])
    for i in range(len(result)):
        for j in range(len(result[0])):
            assert abs(result[i][j] - expected_result[i][j]) < 1e-3


def test_model_train_one_feature():
    feature_matrix = [[1.0]]
    labels = [[1.0]]
    expected_result = train_python_impl(feature_matrix, labels)
    result = train_c_impl(feature_matrix, labels)
    assert len(result) == len(expected_result)
    assert len(result[0]) == len(expected_result[0])
    for i in range(len(result)):
        for j in range(len(result[0])):
            assert abs(result[i][j] - expected_result[i][j]) < 1e-3


def test_model_predict_2_features():
    feature_matrix = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    labels = [[11.0], [12.0], [13.0], [14.0], [15.0]]
    weights = train_python_impl(feature_matrix, labels)
    test_feature_vector = [2.0]
    expected_result = weights[0][0] + weights[1][0] * test_feature_vector[0]
    result = predict_c_impl(test_feature_vector, weights)
    assert abs(result - expected_result) < 1e-3


def test_model_predict_with_regularization_low_lambda():
    feature_matrix = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    labels = [[11.0], [12.0], [13.0], [14.0], [15.0]]
    lambda_l2 = 0.001
    weights = train_python_impl(feature_matrix, labels, lambda_l2)
    test_feature_vector = [2.0]
    expected_result = weights[0][0] + weights[1][0] * test_feature_vector[0]
    result = predict_c_impl(test_feature_vector, weights)
    assert abs(result - expected_result) < 1e-3


def test_model_predict_with_regularization_high_lambda():
    feature_matrix = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    labels = [[11.0], [12.0], [13.0], [14.0], [15.0]]
    lambda_l2 = 100.0
    weights = train_python_impl(feature_matrix, labels, lambda_l2)
    test_feature_vector = [2.0]
    expected_result = weights[0][0] + weights[1][0] * test_feature_vector[0]
    result = predict_c_impl(test_feature_vector, weights)
    assert abs(result - expected_result) < 1e-3

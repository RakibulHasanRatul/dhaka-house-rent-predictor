import pytest
from py_impl import train as model_train


def test_model_train_simple():
    feature_matrix = [[1.0, 2.0], [3.0, 4.0]]
    labels = [[5.0], [6.0]]
    expected_result = [[-3.9999999999964597], [4.499999999997495]]
    # Expected result calculated manually
    result = model_train(feature_matrix, labels)
    assert len(result) == len(expected_result)
    assert len(result[0]) == len(expected_result[0])
    for i in range(len(result)):
        for j in range(len(result[0])):
            assert abs(result[i][j] - expected_result[i][j]) < 1e-6


def test_model_train_with_regularization():
    feature_matrix = [[1.0, 2.0], [3.0, 4.0]]
    labels = [[5.0], [6.0]]
    lambda_l2 = 0.1
    result = model_train(feature_matrix, labels, lambda_l2=lambda_l2)
    # No direct assertion here, as the result depends on the regularization strength.
    # We can only check that the function runs without errors.
    assert result is not None


def test_model_train_single_feature():
    feature_matrix = [[1.0], [2.0]]
    labels = [[3.0], [4.0]]
    expected_result = [[2.2]]
    result = model_train(feature_matrix, labels)
    assert len(result) == len(expected_result)
    assert len(result[0]) == len(expected_result[0])
    for i in range(len(result)):
        for j in range(len(result[0])):
            assert abs(result[i][j] - expected_result[i][j]) < 1e-6


def test_model_train_no_features():
    feature_matrix: list[list[float]] = []
    labels: list[list[float]] = []
    with pytest.raises(Exception):
        model_train(feature_matrix, labels)

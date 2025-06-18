from performance_metrics_functions import mae, mse, r_squared
import pytest


def test_r_squared_perfect_prediction():
    # When predictions are exactly the same as original values
    y_original = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_predicted = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert r_squared(y_predicted, y_original) == pytest.approx(1.0)


def test_r_squared_random_prediction():
    # Test with random values
    y_original = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_predicted = [1.5, 2.5, 3.5, 4.5, 5.5]
    y_mean = sum(y_original) / len(y_original)
    expected = sum((y_p - y_mean) ** 2 for y_p in y_predicted) / sum(
        (y_o - y_mean) ** 2 for y_o in y_original
    )
    assert r_squared(y_predicted, y_original) == pytest.approx(expected)


def test_r_squared_constant_values():
    # When all values are the same
    y_original = [5.0, 5.0, 5.0, 5.0]
    y_predicted = [4.0, 4.0, 4.0, 4.0]
    assert r_squared(y_predicted, y_original) == pytest.approx(0.0)


def test_mse_perfect_prediction():
    # When predictions are exactly the same as original values
    y_original = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_predicted = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert mse(y_predicted, y_original) == pytest.approx(0.0)


def test_mse_random_prediction():
    # Test with random values
    y_original = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_predicted = [1.5, 2.5, 3.5, 4.5, 5.5]
    expected = (0.25 + 0.25 + 0.25 + 0.25 + 0.25) / 5
    assert mse(y_predicted, y_original) == pytest.approx(expected)


def test_mae_perfect_prediction():
    # When predictions are exactly the same as original values
    y_original = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_predicted = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert mae(y_predicted, y_original) == pytest.approx(0.0)


def test_mae_random_prediction():
    # Test with random values
    y_original = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_predicted = [1.5, 2.5, 3.5, 4.5, 5.5]
    expected = (0.5 + 0.5 + 0.5 + 0.5 + 0.5) / 5
    assert mae(y_predicted, y_original) == pytest.approx(expected)


def test_mae_negative_values():
    # Test with negative values
    y_original = [-1.0, -2.0, -3.0, -4.0, -5.0]
    y_predicted = [-1.5, -2.5, -3.5, -4.5, -5.5]
    expected = (0.5 + 0.5 + 0.5 + 0.5 + 0.5) / 5
    assert mae(y_predicted, y_original) == pytest.approx(expected)


def test_length_mismatch():
    # Test that functions raise ValueError when input lists have different lengths
    y_original = [1.0, 2.0, 3.0]
    y_predicted = [1.0, 2.0]

    with pytest.raises(ValueError):
        r_squared(y_predicted, y_original)

    with pytest.raises(ValueError):
        mse(y_predicted, y_original)

    with pytest.raises(ValueError):
        mae(y_predicted, y_original)

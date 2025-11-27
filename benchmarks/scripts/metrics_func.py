def r_squared(y_predicted: list[float], y_original: list[float]) -> float:
    if len(y_original) != len(y_predicted):
        raise ValueError("Length of predicted and original lists must be the same.")

    # in the previous commits, I used the formula r_squared_value = 1 - (ss_res / ss_tot)
    # but to reduced the number of mathematical operations,
    # I used the formula r_squared_value = ss_reg / ss_tot
    # this formula is defined in wikipedia
    # you can learn more at:
    # https://en.wikipedia.org/wiki/Coefficient_of_determination#As_explained_variance

    y_mean = sum(y_original) / len(y_original)

    _q = sum((y_o - y_mean) ** 2 for y_o in y_original)
    if _q == 0:
        # All y values are (almost) the same, R squared value is undefined
        # treating as 0 for safe reporting
        return 0.0

    _p = sum((y_p - y_mean) ** 2 for y_p in y_predicted)

    return _p / _q


def mse(y_predicted: list[float], y_original: list[float]) -> float:
    if len(y_original) != len(y_predicted):
        raise ValueError("Length of predicted and original lists must be the same.")

    return sum((float(y) - float(y_hat)) ** 2 for y, y_hat in zip(y_original, y_predicted)) / len(
        y_original
    )


def mae(y_predicted: list[float], y_original: list[float]) -> float:
    if len(y_original) != len(y_predicted):
        raise ValueError("Length of predicted and original lists must be the same.")

    return sum(abs(float(y) - float(y_hat)) for y, y_hat in zip(y_original, y_predicted)) / len(
        y_original
    )

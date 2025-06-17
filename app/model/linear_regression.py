# vector algebra
# ``` pseudo code
# return inverse(transpose(x).x).transpose(x).label_set_y
# ```

from .matrix_calculation import inverse, matrix_mul, transpose


def model_train(
    feature_matrix: list[list[float]],
    labels: list[float],
    lambda_l2: float = 1e-13,
) -> list[list[float]]:
    y = [[label] for label in labels]

    x = feature_matrix
    xt = transpose(x)
    xtx = matrix_mul(xt, x)

    # regularize
    for r in range(len(xtx[0])):
        xtx[r][r] += lambda_l2

    xtx_inv = inverse(xtx)
    xtx_inv_xt = matrix_mul(xtx_inv, xt)
    return matrix_mul(xtx_inv_xt, y)

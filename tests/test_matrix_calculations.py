import pytest

from model_impls.py_impl.matrix_calculation import (
    construct_identity_matrix,
    inverse,
    matrix_mul,
    transpose,
)


def test_construct_identity_matrix():
    assert construct_identity_matrix(1) == [[1.0]]
    assert construct_identity_matrix(3) == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert construct_identity_matrix(0) == []


def test_inverse():
    matrix1 = [[1.0, 2.0], [3.0, 4.0]]
    inverse1 = inverse(matrix1)
    expected_inverse1 = [[-2.0, 1.0], [1.5, -0.5]]
    assert pytest.approx(inverse1[0][0]) == expected_inverse1[0][0]
    assert pytest.approx(inverse1[0][1]) == expected_inverse1[0][1]
    assert pytest.approx(inverse1[1][0]) == expected_inverse1[1][0]
    assert pytest.approx(inverse1[1][1]) == expected_inverse1[1][1]

    matrix2 = [[2.0, 0.0], [0.0, 2.0]]
    inverse2 = inverse(matrix2)
    expected_inverse2 = [[0.5, 0.0], [0.0, 0.5]]
    assert pytest.approx(inverse2[0][0]) == expected_inverse2[0][0]
    assert pytest.approx(inverse2[0][1]) == expected_inverse2[0][1]
    assert pytest.approx(inverse2[1][0]) == expected_inverse2[1][0]
    assert pytest.approx(inverse2[1][1]) == expected_inverse2[1][1]

    with pytest.raises(ValueError):
        inverse([[0.0, 0.0], [0.0, 0.0]])

    with pytest.raises(ValueError):
        inverse([[1.0, 1.0], [1.0, 1.0]])


def test_transpose():
    matrix1 = [[1.0, 2.0], [3.0, 4.0]]
    transposed1 = transpose(matrix1)
    expected_transposed1 = [[1.0, 3.0], [2.0, 4.0]]
    assert transposed1 == expected_transposed1

    matrix2 = [[1.0, 2.0, 3.0]]
    transposed2 = transpose(matrix2)
    expected_transposed2 = [[1.0], [2.0], [3.0]]
    assert transposed2 == expected_transposed2

    matrix3 = [[1.0], [2.0], [3.0]]
    transposed3 = transpose(matrix3)
    expected_transposed3 = [[1.0, 2.0, 3.0]]
    assert transposed3 == expected_transposed3

    matrix4 = []
    transposed4 = transpose(matrix4)
    assert transposed4 == []

    matrix5 = [[]]
    transposed5 = transpose(matrix5)
    assert transposed5 == [[]]


def test_matrix_mul():
    matrix1 = [[1.0, 2.0], [3.0, 4.0]]
    matrix2 = [[5.0, 6.0], [7.0, 8.0]]
    result1 = matrix_mul(matrix1, matrix2)
    expected_result1 = [[19.0, 22.0], [43.0, 50.0]]
    assert result1 == expected_result1

    matrix3 = [[1.0, 2.0, 3.0]]
    matrix4 = [[4.0], [5.0], [6.0]]
    result2 = matrix_mul(matrix3, matrix4)
    expected_result2 = [[32.0]]
    assert result2 == expected_result2

    with pytest.raises(ValueError):
        matrix_mul([[1.0, 2.0]], [[1.0], [2.0], [3.0]])

    with pytest.raises(ValueError):
        matrix_mul([[1.0, 2.0, 3.0]], [[1.0, 2.0]])

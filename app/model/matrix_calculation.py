def construct_identity_matrix(n: int) -> list[list[float]]:
    identity_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        identity_matrix[i][i] = 1.0
    return identity_matrix


def inverse(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    identity_matrix = construct_identity_matrix(n)
    augmented_matrix = [r[:] + i[:] for r, i in zip(matrix, identity_matrix)]

    # print("Initial augmented_matrix:")
    # print(augmented_matrix)

    for r in range(n):
        pivot = augmented_matrix[r][r]
        if pivot == 0:
            raise ValueError("Matrix is singular, cannot be inverted.")

        for i in range(2 * n):
            augmented_matrix[r][i] /= pivot

        for i in range(n):
            if i != r:
                factor = augmented_matrix[i][r]
                for j in range(2 * n):
                    augmented_matrix[i][j] -= factor * augmented_matrix[r][j]

        # print(f"Augmented matrix after eliminating {r}:")
        # print(augmented_matrix)

    inverse = [row[n:] for row in augmented_matrix]
    return inverse


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    if not matrix:
        return []
    if not matrix[0]:
        return [[]]

    rows = len(matrix)
    cols = len(matrix[0])
    result = [[0.0] * rows for _ in range(cols)]

    for i in range(rows):
        row = matrix[i]
        for j in range(cols):
            result[j][i] = row[j]

    return result


def matrix_mul(mat_a: list[list[float]], mat_b: list[list[float]]) -> list[list[float]]:
    rows_a = len(mat_a)
    cols_a = len(mat_a[0])
    rows_b = len(mat_b)
    cols_b = len(mat_b[0])

    if cols_a != rows_b:
        raise ValueError(
            f"Incompatible matrix dimensions - {rows_a} x {cols_a} and {rows_b} x {cols_b} for multiplication."
        )

    result = [[0.0] * cols_b for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += mat_a[i][k] * mat_b[k][j]

    return result

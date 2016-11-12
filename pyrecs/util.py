import numpy as np

def matrix_to_dataset(matrix):
    X = []
    y = []
    for r, row in enumerate(matrix):
        for c, val in enumerate(row):
            if val is not None and np.isfinite(val):
                X.append((r, c))
                y.append(val)
    return X, y


def dataset_to_matrix(X, y, dtype=None, shape=None):
    X, y = list(X), list(y)

    if shape == None:
        shape = (max(r for r, _ in X) + 1,
                 max(c for _, c in X) + 1)

    matrix = np.ones(shape, dtype=dtype) * np.nan

    for loc, val in zip(X, y):
        if val != None and np.isfinite(val):
            matrix[loc[0]][loc[1]] = val

    return matrix

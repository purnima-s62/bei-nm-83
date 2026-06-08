import numpy as np

A = np.array([
    [1, 1, 1],
    [4, -2, 5],
    [2, -3, 1]
], dtype=float)

B = np.array([9, 29, 3], dtype=float)

n = len(B)
col_index = np.arange(n)

for k in range(n - 1):
    sub = abs(A[k:, k:])
    p, q = np.unravel_index(np.argmax(sub), sub.shape)

    max_row = k + p
    max_col = k + q

    A[[k, max_row]] = A[[max_row, k]]
    B[[k, max_row]] = B[[max_row, k]]

    A[:, [k, max_col]] = A[:, [max_col, k]]
    col_index[[k, max_col]] = col_index[[max_col, k]]

    for i in range(k + 1, n):
        factor = A[i, k] / A[k, k]

        for j in range(k, n):
            A[i, j] -= factor * A[k, j]

        B[i] -= factor * B[k]

x = np.zeros(n)

for i in range(n - 1, -1, -1):
    sum_ax = 0

    for j in range(i + 1, n):
        sum_ax += A[i, j] * x[j]

    x[i] = (B[i] - sum_ax) / A[i, i]

x_final = np.zeros(n)

for i in range(n):
    x_final[col_index[i]] = x[i]

print("Solution:")
print(x_final)
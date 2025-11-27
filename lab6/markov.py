import numpy as np

intensities = {
    (1, 3): 2, (1, 5): 7,
    (2, 1): 4, (2, 3): 14, (2, 4): 2,
    (3, 1): 10, (3, 2): 13, (3, 4): 12, (3, 5): 9,
    (4, 1): 5, (4, 2): 7, (4, 3): 11,
    (5, 1): 6, (5, 2): 7, (5, 3): 8, (5, 4): 6
}

N = 5

Lambda = np.zeros(N)
Lambda[0] = 9  # S1: 2 + 7
Lambda[1] = 20 # S2: 4 + 14 + 2
Lambda[2] = 44 # S3: 10 + 13 + 12 + 9
Lambda[3] = 23 # S4: 5 + 7 + 11
Lambda[4] = 27 # S5: 6 + 7 + 8 + 6

for i in range(N):
    print(f"Λ{i+1} = {Lambda[i]}")

Q = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        state_pair = (i + 1, j + 1)
        if i == j:
            Q[i, j] = -Lambda[i]
        elif state_pair in intensities:
            Q[i, j] = intensities[state_pair]

Q_transposed = Q.T
print(Q)

A = Q_transposed.copy()

A[N-1, :] = 1.0

b = np.zeros(N)
b[N-1] = 1.0

pi_vector = np.linalg.solve(A, b)

print("\nСтационарные Вероятности (pi)")
for i in range(N):
    print(f"π{i+1} = {pi_vector[i]:.4f}")

# проверка
print(f"\nПроверка нормировки (сумма): {np.sum(pi_vector):.4f}")
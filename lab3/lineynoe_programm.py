import numpy as np
from scipy.optimize import linprog

prices_arg = np.array([4, 1, 5, 6, 3.5, 7, 4])

vitamin_content_arg = np.array([
    [5, 0, 2, 0, 3, 1, 2],
    [3, 1, 5, 0, 2, 0, 1],
    [1, 0, 3, 1, 2, 0, 6]
])

requirements_arg = np.array([100, 80, 120])

def vitamin_problem_scipy(prices, vitamin_content, requirements):
    # встроенная функция
    result = linprog(
        c=prices,
        A_ub=-vitamin_content,
        b_ub=[-r for r in requirements],
        bounds=[(0, None)] * 7,
        method='highs'
    )

    if result.success:
        print("Найденное решение с помощью встроенной функции:")
        print(result.message)

        optimal_quantities = result.x
        vitamins = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']

        print("Оптимальное количество поливитаминов:")
        total_amount = 0
        for i, vitamin in enumerate(vitamins):
            amount = optimal_quantities[i]
            if amount > 0:
                print(vitamin + ":", amount, "г")
                total_amount += amount
            else:
                print(vitamin + ": 0.00 г")

        obtained_vitamins = vitamin_content @ optimal_quantities
        vitamin_names = ['Витамин A', 'Витамин C', 'Витамин B6']

        print("Получено витаминов:")
        for i in range(3):
            obtained = obtained_vitamins[i]
            required = requirements[i]
            name = vitamin_names[i]
            print(name + ":", obtained, "из", required, "ед.")

        print("\nОбщее количество поливитаминов:", total_amount, "г")
        print("Количество поливитамина Р1:", optimal_quantities[0], "г")
        print("Минимальные затраты:", result.fun, "тыс. руб.")

    else:
        print("Решение не найдено!")
        print(result.message)

    return result

def simplex_method_two_phase(prices_arg, vitamin_content_arg, requirements_arg):
    m, n = vitamin_content_arg.shape

    vitamin_content1 = np.hstack([vitamin_content_arg, -np.eye(m)])
    requirements1 = requirements_arg.copy()

    tableau = np.zeros((m+1, n+m+1))
    tableau[:m, :n+m] = vitamin_content1
    tableau[:m, -1] = requirements1
    tableau[-1, :n] = 0
    tableau[-1, n:n + m] = 1

    basis = list(range(n, n+m))  # искусственные переменные в базе

    def pivot_on(tableau, row, col):
        pivot_val = tableau[row, col]
        tableau[row, :] /= pivot_val
        for r in range(tableau.shape[0]):
            if r != row:
                tableau[r, :] -= tableau[r, col] * tableau[row, :]

    while True:
        col = np.where(tableau[-1, :-1] < 0)[0]
        if len(col) == 0:
            break
        entering = col[np.argmin(tableau[-1, col])]

        ratios = []
        for i in range(m):
            if tableau[i, entering] > 1e-12:
                ratios.append(tableau[i, -1] / tableau[i, entering])
            else:
                ratios.append(np.inf)
        leaving = np.argmin(ratios)

        if ratios[leaving] == np.inf:
            raise Exception("Задача неограничена")

        pivot_on(tableau, leaving, entering)
        basis[leaving] = entering

    if abs(tableau[-1, -1]) > 1e-5:
        raise Exception("Нет решений")

    tableau = tableau[:, list(range(n)) + [-1]]

    tableau[-1, :-1] = -prices_arg
    for i, bi in enumerate(basis):
        if bi < n:
            tableau[-1, :] += prices_arg[bi] * tableau[i, :]

    while True:
        col = np.where(tableau[-1, :-1] < -1e-12)[0]
        if len(col) == 0:
            break
        entering = col[np.argmin(tableau[-1, col])]

        ratios = []
        for i in range(m):
            if tableau[i, entering] > 1e-12:
                ratios.append(tableau[i, -1] / tableau[i, entering])
            else:
                ratios.append(np.inf)
        leaving = np.argmin(ratios)

        if ratios[leaving] == np.inf:
            raise Exception("Задача неограничена")

        pivot_on(tableau, leaving, entering)
        basis[leaving] = entering

    x = np.zeros(n)
    for i, bi in enumerate(basis):
        if bi < n:
            x[bi] = tableau[i, -1]

    return x, tableau[-1, -1]

# встроенная функция
result = vitamin_problem_scipy(prices_arg, vitamin_content_arg, requirements_arg)

vitamin_optimal, min_cost = simplex_method_two_phase(prices_arg, vitamin_content_arg, requirements_arg)

total_vitamins = np.sum(vitamin_optimal)
vitamin_quantity = vitamin_optimal[0]

print("\nОптимальное количество каждого поливитаминов (г):", vitamin_optimal)
print("Минимальные затраты (тыс.р.):", min_cost)
print("Общее количество витаминов (г):", total_vitamins)
print("Количество поливитамина P1 (г):", vitamin_quantity)




#not finished
import numpy as np
from scipy.optimize import linprog

prices_arg = np.array([4, 1, 5, 6, 3.5, 7, 4])

vitamin_content_arg = np.array([
    [5, 0, 2, 0, 3, 1, 2],
    [3, 1, 5, 0, 2, 0, 1],
    [1, 0, 3, 1, 2, 0, 6]
])

requirements_arg = np.array([100, 80, 120])

# Решение через scipy.linprog
def vitamin_problem_scipy(prices, vitamin_content, requirements):
    result = linprog(
        c=prices,
        A_ub=-vitamin_content,
        b_ub=-requirements,
        bounds=[(0, None)] * len(prices),
        method='highs'
    )

    if result.success:
        print("Решение через linprog:")
        for i, x in enumerate(result.x):
            print(f"P{i + 1}: {x:.2f} г")
        print(f"Минимальные затраты: {result.fun:.2f} тыс. руб.")
    else:
        print("Решение не найдено!")
    return result

result = vitamin_problem_scipy(prices_arg, vitamin_content_arg, requirements_arg)

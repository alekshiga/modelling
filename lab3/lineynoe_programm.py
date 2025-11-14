import numpy as np
from scipy.optimize import linprog

prices = [4, 1, 5, 6, 3.5, 7, 4]

vitamin_content = np.array([
        [5, 0, 2, 0, 3, 1, 2],
        [3, 1, 5, 0, 2, 0, 1],
        [1, 0, 3, 1, 2, 0, 6]
    ])

requirements = [100, 80, 120]

class Simplex:
    def __init__(self, c, A, b):
        # к-нты целевой функции
        self.c = np.array(c, dtype=float)
        # матрица ограничений
        self.A = np.array(A, dtype=float)
        # правые части ограничений
        self.b = np.array(b, dtype=float)
        self.m, self.n = A.shape

    def add_variables(self):
        self.c_max = -self.c

        self.A_negative = -self.A
        self.b_negative = -self.b

        # дополнительные переменные
        self.A_added = np.hstack([self.A_negative, np.eye(self.m)])
        self.c_added = np.hstack([self.c_max, np.zeros(self.m)])

        # искусственные переменные
        self.A_artificial = np.hstack([self.A_added, np.eye(self.m)])

        M = -1e6
        self.c_artificial = np.hstack([self.c_added, np.full(self.m, M)])

    def initialize(self):
        total_vars = self.n + 2 * self.m

        self.tableau = np.zeros((self.m + 1, total_vars + 1))

        # заполняем ограничения
        self.tableau[:self.m, :total_vars] = self.A_artificial
        self.tableau[:self.m, -1] = self.b_negative

        # целевая функция
        self.tableau[-1, :total_vars] = self.c_artificial

        # начальный базис с использованием искуственных переменных
        self.basis = list(range(self.n + self.m, total_vars))

        print("\n\nНачальная симплекс-таблица для самостоятельной реализации:")
        self.print()

    def print(self):
        headers = []
        headers.append(" ")
        for i in range(self.n):
            headers.append(f"      x{i + 1}")
        for i in range(self.m):
            headers.append(f"      s{i + 1}")
        for i in range(self.m):
            headers.append(f"      y{i + 1}")
        headers.append("Решение")

        print("\n" + " | ".join(headers))
        print("-" * (len(headers) * 10))

        for i in range(self.m):
            row_str = " | ".join([f"{val:8.2f}" for val in self.tableau[i]])
            print(f"  | {row_str}")

        # целевая функция
        row_str = " | ".join([f"{val:8.2f}" for val in self.tableau[-1]])
        print(f"F | {row_str}")

        print("\n\n")
        for i in range(self.m):
            row_str = " | ".join([f"{val:8.1f}" for val in self.tableau[i]])
            basis_var = headers[self.basis[i]]
            print(f"{basis_var} | {row_str}")

    def find_pivot_column(self):
        last_row = self.tableau[-1, :-1]

        # исключаем дополнительные и искуственные переменные
        real_vars_indices = list(range(self.n))
        coefficients = last_row[real_vars_indices]

        if all(coefficients <= 1e-10):
            return None  # решение оптимально, если все коэф-ты < 0

        pivot_col = real_vars_indices[np.argmax(coefficients)]
        return pivot_col

    def find_pivot_row(self, pivot_col):
        ratios = []
        for i in range(self.m):
            if self.tableau[i, pivot_col] > 1e-10:
                ratio = self.tableau[i, -1] / self.tableau[i, pivot_col]
                ratios.append(ratio)
            else:
                ratios.append(float('inf'))

        if all(r == float('inf') for r in ratios):
            return None

        pivot_row = np.argmin(ratios)
        return pivot_row

    def pivot(self, pivot_row, pivot_col):
        # поворот таблицы
        pivot_element = self.tableau[pivot_row, pivot_col]

        # делим pivot-строку на pivot элемент
        self.tableau[pivot_row] /= pivot_element

        # обновляем остальные строки путём вычитания pivot-строки
        for i in range(self.m + 1):
            if i != pivot_row:
                multiplier = self.tableau[i, pivot_col]
                self.tableau[i] -= multiplier * self.tableau[pivot_row]

        # обновляем базис
        self.basis[pivot_row] = pivot_col

    def solve(self):

        self.add_variables()
        self.initialize()

        iteration = 0
        max_iterations = 100

        while iteration < max_iterations:
            iteration += 1

            # находим pivot-столбец
            pivot_col = self.find_pivot_column()
            if pivot_col is None:
                print("Достигнут оптимум!")
                break

            # находим pivot-строку
            pivot_row = self.find_pivot_row(pivot_col)
            if pivot_row is None:
                print("Задача неограниченна")
                break

            # выполняем поворот
            self.pivot(pivot_row, pivot_col)
            self.print()

        artificial = False
        solution = np.zeros(self.n)
        for i, basis_var in enumerate(self.basis):
            if basis_var < self.n:
                solution[basis_var] = self.tableau[i, -1]

        optimal_value = -self.tableau[-1, -1]

        return solution, optimal_value

def vitamin_problem_scipy():
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

if __name__ == "__main__":
    # с помощью встроенной функции
    result = vitamin_problem_scipy()

    # самостоятельная реализация
    solver = Simplex(prices, vitamin_content, requirements)
    result_self, optimal_value = solver.solve()

    vitamins = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']
    print("Оптимальное количество поливитаминов:")
    total_amount = 0
    for i, vit in enumerate(vitamins):
        amount = result_self[i]
        print(f"{vit}: {amount}")
        total_amount += amount

    print(f"\nОбщее количество: {total_amount}")
    print(f"Минимальная стоимость: {optimal_value}")

    # Проверка ограничений
    print("\nПроверка ограничений:")
    obtained = vitamin_content @ result_self
    vitamin_names = ['Витамин A', 'Витамин C', 'Витамин B6']

    for i in range(3):
        print(f"{vitamin_names[i]}: {obtained[i]} ≥ {requirements[i]}")
import copy
import math


class TSPSolver:
    def __init__(self, matrix):
        self.original_matrix = copy.deepcopy(matrix)
        self.n = len(matrix)
        self.best_cost = math.inf
        self.best_path = []
        self.steps_log = []

    def reduce_matrix(self, matrix):
        reduction_cost = 0
        n = len(matrix)

        for i in range(n):
            valid_values = [matrix[i][j] for j in range(n) if not math.isinf(matrix[i][j])]
            if not valid_values:
                continue
            min_val = min(valid_values)
            if min_val > 0:
                reduction_cost += min_val
                for j in range(n):
                    if not math.isinf(matrix[i][j]):
                        matrix[i][j] -= min_val

        for j in range(n):
            valid_values = [matrix[i][j] for i in range(n) if not math.isinf(matrix[i][j])]
            if not valid_values:
                continue
            min_val = min(valid_values)
            if min_val > 0:
                reduction_cost += min_val
                for i in range(n):
                    if not math.isinf(matrix[i][j]):
                        matrix[i][j] -= min_val

        return matrix, reduction_cost

    def calculate_penalties(self, matrix):
        n = len(matrix)
        max_penalty = -1
        best_i, best_j = -1, -1

        for i in range(n):
            for j in range(n):
                if matrix[i][j] == 0:
                    row_values = [matrix[i][k] for k in range(n) if k != j and not math.isinf(matrix[i][k])]
                    row_min = min(row_values) if row_values else 0

                    col_values = [matrix[k][j] for k in range(n) if k != i and not math.isinf(matrix[k][j])]
                    col_min = min(col_values) if col_values else 0

                    penalty = row_min + col_min
                    if penalty > max_penalty:
                        max_penalty = penalty
                        best_i, best_j = i, j

        return best_i, best_j, max_penalty

    def exclude_row_col(self, matrix, row, col):
        n = len(matrix)
        new_matrix = copy.deepcopy(matrix)

        for j in range(n):
            new_matrix[row][j] = math.inf
        for i in range(n):
            new_matrix[i][col] = math.inf

        new_matrix[col][row] = math.inf

        return new_matrix

    def solve(self):
        self.steps_log = []
        initial_matrix = copy.deepcopy(self.original_matrix)
        self._solve_recursive(initial_matrix, 0, [], "Начальная матрица")
        return self.best_path, self.best_cost, self.steps_log

    def _solve_recursive(self, matrix, current_cost, current_path, step_info):
        n = self.n

        # логирование
        step_data = {
            'current_cost': current_cost,
            'current_path': copy.deepcopy(current_path),
            'info': step_info
        }
        self.steps_log.append(step_data)

        if len(current_path) == n:
            total_cost = 0
            full_path = current_path + [current_path[0]]

            for k in range(len(full_path) - 1):
                i, j = full_path[k], full_path[k + 1]
                total_cost += self.original_matrix[i][j]

            if total_cost < self.best_cost:
                self.best_cost = total_cost
                self.best_path = current_path
            return
        try:
            reduced_matrix, reduction_cost = self.reduce_matrix(copy.deepcopy(matrix))
        except:
            return

        lower_bound = current_cost + reduction_cost

        # если текущая оценка хуже лучшего решения - отсекаем
        if lower_bound >= self.best_cost:
            return

        i, j, penalty = self.calculate_penalties(reduced_matrix)

        if i == -1:
            return

        include_matrix = self.exclude_row_col(reduced_matrix, i, j)

        if not current_path:
            new_path = [i, j]
        else:
            new_path = current_path + [j]

        self._solve_recursive(
            include_matrix,
            current_cost + reduction_cost,
            new_path,
            f"Включаем ребро ({i + 1},{j + 1}), штраф: {penalty}"
        )

        exclude_matrix = copy.deepcopy(reduced_matrix)
        exclude_matrix[i][j] = math.inf

        self._solve_recursive(
            exclude_matrix,
            current_cost + reduction_cost,
            current_path,
            f"Исключаем ребро ({i + 1},{j + 1})"
        )


def print_matrix(matrix):
    n = len(matrix)
    for i in range(n):
        row_str = ""
        for j in range(n):
            if math.isinf(matrix[i][j]):
                row_str += "   ∞"
            else:
                row_str += f"{matrix[i][j]:>4}"
        print(row_str)


def main():
    # бесконечность вместо исопльзования 1000 и т.д.
    INF = math.inf
    cost_matrix = [
        [INF, 4, 15, 2, 0, 21],
        [25, INF, 13, 19, 15, 6],
        [22, 15, INF, 26, 26, 23],
        [12, 15, 22, INF, 23, 5],
        [9, 6, 18, 21, INF, 6],
        [14, 28, 1, 2, 5, INF]
    ]

    print_matrix(cost_matrix)

    solver = TSPSolver(cost_matrix)
    best_path, best_cost, steps_log = solver.solve()

    if best_path:
        full_path = best_path + [best_path[0]]

        print("Маршрут:".join(str(city + 1) for city in full_path))
        print(f"Минимальная стоимость: {best_cost}")

        print("\nДетали маршрута:")
        total_verify = 0
        for k in range(len(full_path) - 1):
            from_city, to_city = full_path[k], full_path[k + 1]
            cost = cost_matrix[from_city][to_city]
            total_verify += cost
            print(f"  {from_city + 1} → {to_city + 1}: стоимость = {cost}")

        print(f"Проверочная сумма: {total_verify}")
    else:
        print("Решение не найдено!")


if __name__ == "__main__":
    main()
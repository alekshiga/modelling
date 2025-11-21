import copy
import math

# travelling salesman problem
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
            if all(math.isinf(matrix[i][j]) for j in range(n)):
                continue
            min_val = min(matrix[i][j] for j in range(n) if not math.isinf(matrix[i][j]))
            if min_val > 0 and min_val != math.inf:
                reduction_cost += min_val
                for j in range(n):
                    if not math.isinf(matrix[i][j]):
                        matrix[i][j] -= min_val

        for j in range(n):
            column = [matrix[i][j] for i in range(n)]
            if all(math.isinf(x) for x in column):
                continue
            min_val = min(x for x in column if not math.isinf(x))
            if min_val > 0 and min_val != math.inf:
                reduction_cost += min_val
                for i in range(n):
                    if not math.isinf(matrix[i][j]):
                        matrix[i][j] -= min_val

        return matrix, reduction_cost

    # как сильно мы проиграем, если не воспользуемся нулевым элементом
    def calculate_penalties(self, matrix):
        n = len(matrix)
        penalties = []

        for i in range(n):
            for j in range(n):
                if matrix[i][j] == 0:
                    row_min = math.inf
                    for k in range(n):
                        if k != j and not math.isinf(matrix[i][k]):
                            row_min = min(row_min, matrix[i][k])

                    col_min = math.inf
                    for k in range(n):
                        if k != i and not math.isinf(matrix[k][j]):
                            col_min = min(col_min, matrix[k][j])

                    penalty = 0
                    if row_min != math.inf:
                        penalty += row_min
                    if col_min != math.inf:
                        penalty += col_min

                    penalties.append((i, j, penalty))

        return penalties

    def exclude_row_col(self, matrix, row, col):
        n = len(matrix)
        new_matrix = copy.deepcopy(matrix)

        # запрещаем преждевременный возврат в исходную точку
        if row < n and col < n:
            new_matrix[col][row] = math.inf

        for i in range(n):
            new_matrix[i][col] = math.inf
        for j in range(n):
            new_matrix[row][j] = math.inf

        return new_matrix

    def solve(self):
        self.steps_log = []
        initial_matrix = copy.deepcopy(self.original_matrix)
        self._solve_recursive(initial_matrix, 0, [], "Начальная матрица")
        return self.best_path, self.best_cost, self.steps_log

    def _solve_recursive(self, matrix, current_cost, current_path, step_info):
        n = len(matrix)

        # логирование
        step_data = {
            'matrix': copy.deepcopy(matrix),
            'current_cost': current_cost,
            'current_path': copy.deepcopy(current_path),
            'info': step_info
        }
        self.steps_log.append(step_data)

        if len(current_path) == n - 1:
            # находим последнее ребро для замыкания маршрута
            used_rows = set(p[0] for p in current_path)
            used_cols = set(p[1] for p in current_path)

            last_row = [i for i in range(n) if i not in used_rows][0]
            last_col = [j for j in range(n) if j not in used_cols][0]

            final_path = current_path + [(last_row, last_col)]

            # вычисляем полную стоимость
            first_city = final_path[0][0]
            last_city = final_path[-1][1]
            final_cost = current_cost + self.original_matrix[last_row][last_col] + self.original_matrix[last_city][
                first_city]

            if final_cost < self.best_cost:
                self.best_cost = final_cost
                self.best_path = final_path

        reduced_matrix, reduction_cost = self.reduce_matrix(matrix)
        lower_bound = current_cost + reduction_cost

        # если текущая оценка хуже лучшего решения - отсекаем
        if lower_bound >= self.best_cost:
            return

        penalties = self.calculate_penalties(reduced_matrix)
        if not penalties:
            return

        max_penalty_edge = max(penalties, key=lambda x: x[2])
        i, j, penalty = max_penalty_edge

        include_matrix = self.exclude_row_col(reduced_matrix, i, j)
        include_cost = current_cost + reduction_cost

        self._solve_recursive(
            include_matrix,
            include_cost,
            current_path + [(i, j)],
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

    print("\nДетали маршрута:")
    total_cost = 0
    for i, (from_city, to_city) in enumerate(best_path[:-1]):
        cost = solver.original_matrix[from_city][to_city]
        total_cost += cost
        print(f"  {from_city + 1} → {to_city + 1}: стоимость = {cost}")

    for i, step in enumerate(steps_log[:10]):
        print(f"\nШаг {i + 1}: {step['info']}")
        print(f"Текущая стоимость: {step['current_cost']}")
        if step['current_path']:
            path = " → ".join(f"({f + 1},{t + 1})" for f, t in step['current_path'])
            print(f"Текущий путь: {path}")

    path_str = " → ".join(str(city + 1) for city, _ in best_path[:-1])
    print(f"Оптимальный маршрут: {path_str}")
    print(f"Минимальная стоимость: {best_cost}")

if __name__ == "__main__":
    main()
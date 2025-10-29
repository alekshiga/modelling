import numpy as np
from collections import deque


class TransportProblem:
    def __init__(self, supplies, demands, costs):
        self.supplies = supplies.copy()
        self.demands = demands.copy()
        self.costs = costs.copy()
        self.n = len(supplies)
        self.m = len(demands)
        self.plan = np.zeros((self.n, self.m))

    def balance_problem(self):
        total_supply = sum(self.supplies)
        total_demand = sum(self.demands)

        if total_supply > total_demand:
            self.demands.append(total_supply - total_demand)
            self.costs = np.hstack((self.costs, np.zeros((self.n, 1))))
            print(f"Добавлен фиктивный потребитель с потребностью: {total_supply - total_demand}")
        elif total_supply < total_demand:
            self.supplies.append(total_demand - total_supply)
            self.costs = np.vstack((self.costs, np.zeros((1, self.m))))
            print(f"Добавлен фиктивный поставщик с запасом: {total_demand - total_supply}")

        self.n = len(self.supplies)
        self.m = len(self.demands)
        self.plan = np.zeros((self.n, self.m))
        return True

    def northwest_corner_method(self):
        print("\nПостроение начального плана с помощью метода Северо-Западного угла")

        plan = np.zeros((self.n, self.m))
        supplies_rest = self.supplies.copy()
        demands_rest = self.demands.copy()

        i, j = 0, 0
        while i < self.n and j < self.m:
            if supplies_rest[i] < demands_rest[j]:
                plan[i][j] = supplies_rest[i]
                demands_rest[j] -= supplies_rest[i]
                supplies_rest[i] = 0
                i += 1
            else:
                plan[i][j] = demands_rest[j]
                supplies_rest[i] -= demands_rest[j]
                demands_rest[j] = 0
                j += 1

        self.plan = plan
        return plan

    def calculate_potentials(self, basis):
        u = [None] * self.n
        v = [None] * self.m

        u[0] = 0

        changed = True
        while changed:
            changed = False
            for i in range(self.n):
                for j in range(self.m):
                    if basis[i][j]:
                        if u[i] is not None and v[j] is None:
                            v[j] = self.costs[i][j] - u[i]
                            changed = True
                        elif v[j] is not None and u[i] is None:
                            u[i] = self.costs[i][j] - v[j]
                            changed = True
        return u, v

    def calculate_deltas(self, u, v, basis):
        deltas = np.zeros((self.n, self.m))
        for i in range(self.n):
            for j in range(self.m):
                if not basis[i][j] and u[i] is not None and v[j] is not None:
                    deltas[i][j] = self.costs[i][j] - (u[i] + v[j])
        return deltas

    def find_best_improvement_cell(self, deltas, basis):
        min_delta = 0
        best_i, best_j = -1, -1

        for i in range(self.n):
            for j in range(self.m):
                if not basis[i][j] and deltas[i][j] < min_delta:
                    min_delta = deltas[i][j]
                    best_i, best_j = i, j

        return best_i, best_j, min_delta

    def find_cycle(self, start_i, start_j, basis):
        temp_basis = basis.copy()
        temp_basis[start_i][start_j] = True

        visited = set()
        path = []

        def dfs(i, j, from_row):
            if (i, j) in visited:
                if (i, j) == (start_i, start_j) and len(path) >= 3:
                    return path + [(i, j)]
                return None

            visited.add((i, j))
            path.append((i, j))

            if from_row:
                for col in range(self.m):
                    if temp_basis[i][col] and col != j:
                        result = dfs(i, col, False)
                        if result:
                            return result
            else:
                for row in range(self.n):
                    if temp_basis[row][j] and row != i:
                        result = dfs(row, j, True)
                        if result:
                            return result

            path.pop()
            visited.remove((i, j))
            return None

        result = dfs(start_i, start_j, True)
        if result and len(result) >= 4:
            return result
        return None

    def improve_plan_with_cycle(self, cycle):
        if not cycle or len(cycle) < 4:
            return False

        # Определяем порядок вершин: начинаем с улучшающей клетки
        start_index = 0
        for idx, (i, j) in enumerate(cycle):
            if idx < len(cycle) - 1 and (i, j) == cycle[0]:
                start_index = idx
                break

        # Переупорядочиваем цикл так, чтобы начинался и заканчивался улучшающей клеткой
        reordered_cycle = cycle[start_index:-1] + cycle[:start_index] + [cycle[start_index]]

        # Находим минимальное значение в отрицательных вершинах (четные индексы, начиная с 1)
        theta = float('inf')
        for idx in range(1, len(reordered_cycle), 2):
            i, j = reordered_cycle[idx]
            if self.plan[i][j] < theta:
                theta = self.plan[i][j]

        if theta == float('inf') or theta <= 0:
            return False

        print(f"θ = {theta}")

        # Перераспределяем груз
        for idx, (i, j) in enumerate(reordered_cycle[:-1]):
            if idx % 2 == 0:  # положительные вершины
                self.plan[i][j] += theta
            else:  # отрицательные вершины
                self.plan[i][j] -= theta

        return True

    def calculate_total_cost(self):
        return np.sum(self.plan * self.costs)

    def solve(self):
        print("\nРешение транспортной задачи")

        total_supply = sum(self.supplies)
        total_demand = sum(self.demands)
        print(f"Сумма запасов: {total_supply}")
        print(f"Сумма потребностей: {total_demand}")

        if total_supply != total_demand:
            print("Задача несбалансированна, выполняется балансировка.")
            self.balance_problem()
        else:
            print("Задача сбалансированна")

        initial_plan = self.northwest_corner_method()
        print("Начальный план перевозок:")
        print(initial_plan)
        print(f"Начальная стоимость: {self.calculate_total_cost():.0f}")

        self._check_constraints("Начальный план")

        iteration = 1
        max_iterations = 50

        while iteration <= max_iterations:
            print(f"\nИтерация №{iteration}")

            basis = self.plan > 0
            basis_cells_count = np.sum(basis)
            print(f"Базисных клеток: {basis_cells_count}")

            if basis_cells_count < self.n + self.m - 1:
                print("План вырожденный, требуется добавить базисную клетку")

            u, v = self.calculate_potentials(basis)
            print(f"Потенциалы u: {[f'{x:.1f}' if x is not None else 'None' for x in u]}")
            print(f"Потенциалы v: {[f'{x:.1f}' if x is not None else 'None' for x in v]}")

            deltas = self.calculate_deltas(u, v, basis)

            print("Матрица оценок Δ:")
            for i in range(self.n):
                row = []
                for j in range(self.m):
                    if basis[i][j]:
                        row.append("  баз  ")
                    else:
                        row.append(f"{deltas[i][j]:6.1f}")
                print(" | ".join(row))

            best_i, best_j, min_delta = self.find_best_improvement_cell(deltas, basis)

            if min_delta >= -1e-10:
                print("✓ Все оценки неотрицательные - план оптимален!")
                break

            print(f"Найдена клетка для улучшения: ({best_i}, {best_j}) с оценкой Δ = {min_delta:.2f}")

            cycle = self.find_cycle(best_i, best_j, basis)

            if cycle is None:
                print("Не удалось найти цикл пересчета")
                break

            print(f"Найден цикл из {len(cycle)} вершин: {cycle}")

            if self.improve_plan_with_cycle(cycle):
                print("План улучшен:")
                print(self.plan)
                print(f"Новая стоимость: {self.calculate_total_cost():.0f}")

                if not self._check_constraints(f"После итерации {iteration}"):
                    print("Error! нарушены ограничения!")
                    break
            else:
                print("Не удалось улучшить план")
                break

            iteration += 1

        if iteration > max_iterations:
            print("Достигнуто максимальное количество итераций")

        return self.plan

    def _check_constraints(self, stage_name):
        print(f"\nПроверка ограничений ({stage_name}):")

        valid = True

        for i in range(self.n):
            sent = np.sum(self.plan[i])
            supply = self.supplies[i]
            if abs(sent - supply) > 1e-10:
                print(f"Error! Поставщик {i + 1}: отправлено {sent:.1f}, должно быть {supply}")
                valid = False
            else:
                print(f"OK! Поставщик {i + 1}: отправлено {sent:.1f} из {supply}")

        for j in range(self.m):
            received = np.sum(self.plan[:, j])
            demand = self.demands[j]
            if abs(received - demand) > 1e-10:
                print(f"Error! Потребитель {j + 1}: получено {received:.1f}, должно быть {demand}")
                valid = False
            else:
                print(f"OK! Потребитель {j + 1}: получено {received:.1f} из {demand}")

        if np.any(self.plan < -1e-10):
            print("Error! найдены отрицательные перевозки")
            valid = False
        else:
            print("OK! Все перевозки неотрицательны")

        return valid

    def print_solution(self):
        print("\nОптимальное решение:")

        total_cost = self.calculate_total_cost()
        print("Поставщик → Потребитель | Количество | Стоимость за ед. | Общая стоимость")

        for i in range(self.n):
            for j in range(self.m):
                if self.plan[i][j] > 1e-10:
                    unit_cost = self.costs[i][j]
                    total_cell_cost = self.plan[i][j] * unit_cost
                    print(
                        f"  {i + 1} → {j + 1}                 | {self.plan[i][j]:8.0f}   | {unit_cost:8.0f}         | {total_cell_cost:8.0f}")

        print(f"Общая стоимость перевозок: {total_cost:.0f} д.е.")

        self._check_constraints("Проверка")


def main():
    supplies = [60, 80, 106]
    demands = [44, 70, 50, 82]

    distances = np.array([
        [13, 17, 6, 8],
        [2, 7, 10, 41],
        [12, 18, 2, 22]
    ])

    costs = distances * 10

    print("Исходные данные:")
    print(f"Запасы на складах: {supplies}")
    print(f"Потребности магазинов: {demands}")
    print("\nМатрица расстояний (км):")
    print(distances)

    problem = TransportProblem(supplies, demands, costs)
    optimal_plan = problem.solve()
    problem.print_solution()


if __name__ == "__main__":
    main()
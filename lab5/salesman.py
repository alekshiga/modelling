import copy
import math


class TSPSolver:
    def __init__(self, matrix):
        self.n = len(matrix)
        self.matrix = matrix
        self.best_cost = math.inf
        self.best_path = []

    def reduce_matrix(self, matrix, current_cost):
        reduction = 0
        for i in range(len(matrix)):
            row_min = min(matrix[i])
            if row_min != math.inf:
                reduction += row_min
                for j in range(len(matrix)):
                    if matrix[i][j] != math.inf:
                        matrix[i][j] -= row_min

        for j in range(len(matrix)):
            col_min = min(matrix[i][j] for i in range(len(matrix)))
            if col_min != math.inf:
                reduction += col_min
                for i in range(len(matrix)):
                    if matrix[i][j] != math.inf:
                        matrix[i][j] -= col_min

        return current_cost + reduction

    def solve(self):
        initial_matrix = copy.deepcopy(self.matrix)
        initial_cost = self.reduce_matrix(initial_matrix, 0)
        self.matrix = initial_matrix
        self.little_algorithm(initial_matrix, initial_cost, [0], 0)
        return self.best_path, self.best_cost

    def little_algorithm(self, matrix, current_cost, path, current_city):
        if len(path) == self.n:
            final_cost = current_cost + self.matrix[path[-1]][0]
            if final_cost < self.best_cost:
                self.best_cost = final_cost
                self.best_path = path + [0]
            return

        if current_cost >= self.best_cost:
            return

        for next_city in range(self.n):
            if next_city not in path:
                new_matrix = copy.deepcopy(matrix)

                # запрещаем переход в города, которые уже посетили
                for i in range(len(new_matrix)):
                    new_matrix[current_city][i] = math.inf
                    new_matrix[i][next_city] = math.inf

                # запрещаем переходы, создающие подциклы
                new_matrix[next_city][0] = math.inf
                transition_cost = matrix[current_city][next_city]
                new_cost = self.reduce_matrix(new_matrix, current_cost + transition_cost)
                self.little_algorithm(new_matrix, new_cost, path + [next_city], next_city)


def main():
    INF = math.inf
    matrix = [
        [INF, 4, 15, 2, 0, 21],
        [25, INF, 13, 19, 15, 6],
        [22, 15, INF, 26, 26, 23],
        [12, 15, 22, INF, 23, 5],
        [9, 6, 18, 21, INF, 6],
        [14, 28, 1, 2, 5, INF]
    ]

    solver = TSPSolver(matrix)
    best_path, best_cost = solver.solve()

    for i in range(len(best_path)):
        best_path[i] += 1

    print("Решение методом Литтла (метод ветвей и границ):")
    print(f"Оптимальный маршрут: {' -> '.join(map(str, best_path))}")
    print(f"Минимальная стоимость: {best_cost}")

if __name__ == "__main__":
    main()
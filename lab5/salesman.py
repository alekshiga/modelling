import copy
import math

import networkx as nx
from matplotlib import pyplot as plt


class TSPSolver:
    def __init__(self, matrix):
        self.n = len(matrix)
        self.matrix = matrix
        self.best_cost = math.inf
        self.best_path = []
        self.steps = []
        self.current_step = 0

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

    def little_algorithm(self, matrix, current_cost, path, current_city, level=0):
        self.current_step += 1
        self.steps.append({
            'step': self.current_step,
            'path': path.copy(),
            'current_cost': current_cost,
            'best_cost': self.best_cost,
            'level': level,
            'matrix': copy.deepcopy(matrix)
        })
        if len(path) == self.n:
            final_cost = current_cost + self.matrix[path[-1]][0]
            if final_cost < self.best_cost:
                self.best_cost = final_cost
                self.best_path = path + [0]
                self.steps.append({
                    'step': self.current_step,
                    'path': self.best_path.copy(),
                    'current_cost': final_cost,
                    'best_cost': self.best_cost,
                    'level': level,
                    'matrix': copy.deepcopy(matrix),
                    'is_solution': True
                })
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
                self.little_algorithm(new_matrix, new_cost, path + [next_city], next_city, level + 1)

    def visualize(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # граф с городами
        G = nx.DiGraph()
        cities = [f'{i}' for i in range(self.n)]
        G.add_nodes_from(cities)

        def update_visualization(step_data):
            ax1.clear()
            ax2.clear()

            current_path = step_data['path']

            pos = nx.circular_layout(G)

            nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='red',
                                   node_size=150, alpha=0.9)

            # текущая итерация
            if len(current_path) > 1:
                current_edges = []
                for i in range(len(current_path) - 1):
                    from_city = current_path[i]
                    to_city = current_path[i + 1]
                    current_edges.append((cities[from_city], cities[to_city]))

                nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=current_edges,
                                       edge_color='black', width=2, alpha=0.8,
                                       arrows=True, arrowsize=20)

            nx.draw_networkx_labels(G, pos, ax=ax1, font_weight='bold')

            edge_labels = {}
            if len(current_path) > 1:
                for i in range(len(current_path) - 1):
                    from_city = current_path[i]
                    to_city = current_path[i + 1]
                    edge_labels[(cities[from_city], cities[to_city])] = self.matrix[from_city][to_city]

            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax1)

            ax1.set_title(f"Шаг {step_data['step']}: Путь {[x + 1 for x in current_path]}\n"
                          f"Текущая стоимость: {step_data['current_cost']:.1f}, "
                          f"Лучшая стоимость: {self.best_cost:.1f}")
            ax1.axis('off')

            # дерево поиска
            levels = {}
            for step in self.steps:
                level = step.get('level', 0)
                if level not in levels:
                    levels[level] = []
                levels[level].append(step)

            for level, steps in levels.items():
                x_positions = [i for i in range(len(steps))]
                y_positions = [level] * len(steps)
                costs = [s['current_cost'] for s in steps]

                colors = ['green' if s.get('is_solution', False) else
                          'red' if s['current_cost'] >= self.best_cost else
                          'blue' for s in steps]

                ax2.scatter(x_positions, y_positions, c=colors, s=25)

                for i, (x, y, cost) in enumerate(zip(x_positions, y_positions, costs)):
                    ax2.text(x, y + 0.1, f'{cost:.1f}', ha='center', fontsize=6)  # Уменьшили шрифт

            ax2.set_xlabel('Ветвь')
            ax2.set_ylabel('Глубина поиска')
            ax2.set_title('Дерево поиска методом ветвей и границ')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.pause(0.001)

        # Анимируем процесс поиска
        print("Запуск анимации поиска...")
        for i, step in enumerate(self.steps):
            update_visualization(step)
            if step.get('is_solution', False):
                print(f"Найдено улучшение на шаге {step['step']}: стоимость {step['current_cost']}")

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

    solver.visualize()

    print("Решение методом Литтла (метод ветвей и границ):")
    print(f"Оптимальный маршрут: {' -> '.join(map(str, best_path))}")
    print(f"Минимальная стоимость: {best_cost}")

if __name__ == "__main__":
    main()
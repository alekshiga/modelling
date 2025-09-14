import pandas as pd
import matplotlib.pyplot as plt


def generate_permutations_recursive(elements, current_permutation, result):
    # рекурсивно генерирует все перестановки списка элементов
    if not elements:
        result.append(current_permutation)
        return

    for i in range(len(elements)):
        new_elements = elements[:i] + elements[i + 1:]
        new_permutation = current_permutation + [elements[i]]
        generate_permutations_recursive(new_elements, new_permutation, result)


def calculate_times_nx3(time):
    # рассчитывает время окончания обработки и простои для nx3
    n = len(time)
    a_times = [time[0] for time in time]
    b_times = [time[1] for time in time]
    c_times = [time[2] for time in time]

    completion_a = [0] * n
    completion_b = [0] * n
    completion_c = [0] * n

    completion_a[0] = a_times[0]
    completion_b[0] = completion_a[0] + b_times[0]
    completion_c[0] = completion_b[0] + c_times[0]

    for i in range(1, n):
        completion_a[i] = completion_a[i - 1] + a_times[i]
        completion_b[i] = max(completion_a[i], completion_b[i - 1]) + b_times[i]
        completion_c[i] = max(completion_b[i], completion_c[i - 1]) + c_times[i]

    total_time = completion_c[-1]

    return total_time, completion_a, completion_b, completion_c


def plot_gantt_chart_nx3(time, machine_times, title):
    # строит график Ганта для задачи nx3
    fig, ax = plt.subplots(figsize=(12, 6))
    machines = ['Станок A', 'Станок B', 'Станок C']
    colors = plt.cm.get_cmap('tab10', len(time))

    for i, time in enumerate(time):
        original_index = time[3]

        # станок A
        start_a = machine_times['A'][i] - time[0]
        ax.broken_barh([(start_a, time[0])], (10, 9), facecolors=colors(original_index - 1), edgecolor='black')
        ax.text(start_a + time[0] / 2, 14.5, f'Деталь {original_index}', ha='center', va='center', color='white',
                fontsize=10)

        # станок B
        start_b = machine_times['B'][i] - time[1]
        ax.broken_barh([(start_b, time[1])], (20, 9), facecolors=colors(original_index - 1), edgecolor='black')
        ax.text(start_b + time[1] / 2, 24.5, f'Деталь {original_index}', ha='center', va='center', color='white',
                fontsize=10)

        # станок C
        start_c = machine_times['C'][i] - time[2]
        ax.broken_barh([(start_c, time[2])], (30, 9), facecolors=colors(original_index - 1), edgecolor='black')
        ax.text(start_c + time[2] / 2, 34.5, f'Деталь {original_index}', ha='center', va='center', color='white',
                fontsize=10)

    ax.set_yticks([14.5, 24.5, 34.5])
    ax.set_yticklabels(machines)
    ax.set_xlabel('Время')
    ax.set_title(title)
    ax.grid(True)
    plt.show()


def brute_force_solver_custom(times):
    # решает задачу методом полного перебора, используя перестановки
    n = len(times)
    indices = list(range(n))
    all_permutations = []

    generate_permutations_recursive(indices, [], all_permutations)

    min_time = float('inf')
    optimal_time = None

    print(f"Количество перестановок для n={n}: {len(all_permutations)}")

    for p in all_permutations:
        current_time = [times[i] for i in p]
        total_time, _, _, _ = calculate_times_nx3(current_time)

        if total_time < min_time:
            min_time = total_time
            optimal_time = current_time

    return optimal_time, min_time


def main():
    # данные, которые не удовлетворяют условию Джонсона
    initial_times = [
        (3, 8, 4, 1),
        (6, 1, 7, 2),
        (2, 5, 9, 3),
        (5, 7, 3, 4)
    ]

    print("Исходные данные:")
    df_initial = pd.DataFrame([(time[3], time[0], time[1], time[2]) for time in initial_times],
                              columns=['№', 'a_i', 'b_i', 'c_i'])
    print(df_initial.to_string(index=False))
    print("-" * 30)

    # расчет времени для исходной последовательности
    initial_total_time, ca_init, cb_init, cc_init = calculate_times_nx3(initial_times)
    print(f"Общее время для исходной последовательности: {initial_total_time}")
    plot_gantt_chart_nx3(initial_times, {'A': ca_init, 'B': cb_init, 'C': cc_init},
                         "График Ганта для исходной последовательности")

    # поиск оптимальной последовательности методом перебора
    optimal_time, optimal_total_time = brute_force_solver_custom(initial_times)

    if optimal_time:
        print("\nОптимальная последовательность запуска:")
        df_optimal = pd.DataFrame([(time[3], time[0], time[1], time[2]) for time in optimal_time],
                                  columns=['№', 'a_i', 'b_i', 'c_i'])
        print(df_optimal.to_string(index=False))
        print("-" * 30)

        # расчет времени для оптимальной последовательности
        optimal_total_time, ca_opt, cb_opt, cc_opt = calculate_times_nx3(optimal_time)
        print(f"Оптимальное время: {optimal_total_time}")
        plot_gantt_chart_nx3(optimal_time, {'A': ca_opt, 'B': cb_opt, 'C': cc_opt},
                             "График Ганта для оптимальной последовательности")


if __name__ == "__main__":
    main()
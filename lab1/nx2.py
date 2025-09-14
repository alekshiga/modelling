import matplotlib.pyplot as plt
import pandas as pd

def johnson(times):
    # сортирует детали по времени обработки
    time_A = sorted([time for time in times if times[0] <= times[1]], key=lambda  x: x[0])
    time_B = sorted([time for time in times if times[0] > times[1]], key = lambda x: x[1], reverse=True)

    optimal_time = time_A + time_B
    return optimal_time

def calculate_times(sequence):
    # рассчитывает время окончания обработки и простои.
    n = len(sequence)
    a = [time[0] for time in sequence]
    b = [time[1] for time in sequence]

    completion_a = [0] * n
    completion_b = [0] * n
    afk_b = [0] * n # время простоя станка B

    completion_a[0] = a[0]
    completion_b[0] = a[0] + b[0]

    for i in range(1, n):
        completion_a[i] = completion_a[i - 1] + a[i]
        completion_b[i] = max(completion_a[i], completion_b[i - 1]) + b[i]
        afk_b[i] = completion_b[i] - completion_b[i - 1] - b[i]

    total_time = completion_b[-1]
    total_idle_b = sum(afk_b)

    return total_time, total_idle_b, completion_a, completion_b, afk_b


def plot_gantt_chart(sequence, machine_times, title):
    # строит график Ганта.
    fig, ax = plt.subplots()
    machines = ['A', 'B']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, job in enumerate(sequence):
        original_index = job[2]

        # станок A
        start_a = machine_times['A'][i] - job[0]
        ax.broken_barh([(start_a, job[0])], (10, 9), facecolors=colors[original_index - 1], edgecolor='black')
        ax.text(start_a + job[0] / 2, 14.5, f'Деталь {original_index}', ha='center', va='center', color='white',
                fontsize=10)

        # станок B
        start_b = machine_times['B'][i] - job[1]
        ax.broken_barh([(start_b, job[1])], (20, 9), facecolors=colors[original_index - 1], edgecolor='black')
        ax.text(start_b + job[1] / 2, 24.5, f'Деталь {original_index}', ha='center', va='center', color='white',
                fontsize=10)

        # простой станка B
        if start_b > machine_times['A'][i]:
            idle_start = machine_times['A'][i]
            idle_duration = start_b - idle_start
            ax.broken_barh([(idle_start, idle_duration)], (20, 9), facecolors='lightgray', edgecolor='black')
            ax.text(idle_start + idle_duration / 2, 24.5, 'Простой', ha='center', va='center', fontsize=8,
                    color='black')

    ax.set_yticks([14.5, 24.5])
    ax.set_yticklabels(['Станок A', 'Станок B'])
    ax.set_xlabel('Время')
    ax.set_title(title)
    ax.grid(True)
    plt.show()

def main():
    initial_times = [
        (11, 2, 1),
        (8, 1, 2),
        (7, 3, 3),
        (2, 9, 4),
        (5, 5, 5)
    ]

    print("Исходные данные:")
    df_initial = pd.DataFrame([(time[2], time[0], time[1]) for time in initial_times], columns=['№', 'a_i', 'b_i'])
    print(df_initial.to_string(index=False))
    print("-" * 30)

    # расчет времени для исходной последовательности
    initial_total_time, initial_idle_b, completion_a, completion_b, _ = calculate_times(initial_times)
    print(f"Общее время для исходной последовательности: {initial_total_time}")
    print(f"Суммарный простой станка B: {initial_idle_b}")
    plot_gantt_chart(initial_times, {'A': completion_a, 'B': completion_b},
                     "График Ганта для исходной последовательности")

    # поиск оптимальной последовательности с помощью алгоритма Джонсона
    optimal_sequence = johnson(initial_times)

    print("\nОптимальная последовательность запуска:")
    df_optimal = pd.DataFrame([(time[2], time[0], time[1]) for time in optimal_sequence], columns=['№', 'a_i', 'b_i'])
    print(df_optimal.to_string(index=False))
    print("-" * 30)

    # расчет времени для оптимальной последовательности
    optimal_total_time, optimal_idle_b, completion_a_opt, completion_b_opt, _ = calculate_times(optimal_sequence)
    print(f"Оптимальное время: {optimal_total_time}")
    print(f"Суммарный простой станка B: {optimal_idle_b}")
    plot_gantt_chart(optimal_sequence, {'A': completion_a_opt, 'B': completion_b_opt},
                     "График Ганта для оптимальной последовательности")


if __name__ == "__main__":
    main()
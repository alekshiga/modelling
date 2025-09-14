import pandas as pd
import matplotlib.pyplot as plt

def johnson_nx3(times):
    # решает задачу Джонсона nx3, сводя ее к nx2.
    a = [time[0] for time in times]
    b = [time[1] for time in times]
    c = [time[2] for time in times]

    # проверка условия Джонсона
    if not (min(a) >= max(b) or min(c) >= max(b)):
        print("Условие Джонсона не выполняется. Для решения необходим полный перебор.")
        return None

    print("Условие Джонсона выполняется. Задача сводится к nx2.")

    # создание "виртуальных" станков A' и B'
    new_times = []
    for time in times:
        a_prime = time[0] + time[1]
        b_prime = time[1] + time[2]
        new_times.append((a_prime, b_prime, time[3]))

    # применение алгоритма Джонсона для nx2
    time_A = sorted([time for time in new_times if time[0] <= time[1]], key=lambda x: x[0])
    time_B = sorted([time for time in new_times if time[0] > time[1]], key=lambda x: x[1], reverse=True)

    optimal_time_prime = time_A + time_B

    # восстановление исходной последовательности
    optimal_time = [times[time[2] - 1] for time in optimal_time_prime]
    return optimal_time


def calculate_times_nx3(time):
    # рассчитывает время окончания обработки и простои для nx3.
    n = len(time)
    a_times = [time[0] for time in time]
    b_times = [time[1] for time in time]
    c_times = [time[2] for time in time]

    completion_a = [0] * n
    completion_b = [0] * n
    completion_c = [0] * n

    # первая деталь
    completion_a[0] = a_times[0]
    completion_b[0] = completion_a[0] + b_times[0]
    completion_c[0] = completion_b[0] + c_times[0]

    # остальные детали
    for i in range(1, n):
        completion_a[i] = completion_a[i - 1] + a_times[i]
        completion_b[i] = max(completion_a[i], completion_b[i - 1]) + b_times[i]
        completion_c[i] = max(completion_b[i], completion_c[i - 1]) + c_times[i]

    total_time = completion_c[-1]

    return total_time, completion_a, completion_b, completion_c


def plot_gantt_chart_nx3(time, machine_times, title):
    # строит график Ганта для задачи nx3.
    fig, ax = plt.subplots(figsize=(12, 6))
    machines = ['Станок A', 'Станок B', 'Станок C']
    colors = plt.cm.get_cmap('tab10', len(time))

    for i, time in enumerate(time):
        original_index = time[3]

        # станция A
        start_a = machine_times['A'][i] - time[0]
        ax.broken_barh([(start_a, time[0])], (10, 9), facecolors=colors(original_index - 1), edgecolor='black')
        ax.text(start_a + time[0] / 2, 14.5, f'Деталь {original_index}', ha='center', va='center', color='white',
                fontsize=10)

        # станция B
        start_b = machine_times['B'][i] - time[1]
        ax.broken_barh([(start_b, time[1])], (20, 9), facecolors=colors(original_index - 1), edgecolor='black')
        ax.text(start_b + time[1] / 2, 24.5, f'Деталь {original_index}', ha='center', va='center', color='white',
                fontsize=10)

        # станция C
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


def main():
    # пример данных, где условие Джонсона выполняется
    initial_times = [
        (11, 6, 7, 1),
        (8, 4, 9, 2),
        (9, 3, 8, 3),
        (10, 2, 12, 4),
        (5, 5, 10, 5)
    ]

    # вывод исходных данных
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

    # поиск оптимальной последовательности
    optimal_time = johnson_nx3(initial_times)

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
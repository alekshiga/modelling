import math
import random
from itertools import permutations

# вычисляет общее время обработки всех деталей
def calculate_total_time(sequence, times):
    n = len(sequence)
    m = len(times[0])

    completion_times = [[0] * m for _ in range(n)]

    completion_times[0][0] = times[sequence[0] - 1][0]
    for j in range(1, m):
        completion_times[0][j] = completion_times[0][j - 1] + times[sequence[0] - 1][j]

    for i in range(1, n):
        for j in range(m):
            t_ij = times[sequence[i] - 1][j]
            t_prev_job = completion_times[i - 1][j]
            t_prev_machine = completion_times[i][j - 1] if j > 0 else 0
            completion_times[i][j] = t_ij + max(t_prev_job, t_prev_machine)

    # общее время обработки
    total_time = completion_times[n - 1][m - 1]
    # общее время простоя станков
    total_idle_time = 0
    # общее время ожидания деталей
    total_wait_time = 0

    for j in range(m):
        idle_time_j = completion_times[n - 1][j] - sum(times[sequence[i] - 1][j] for i in range(n))
        total_idle_time += idle_time_j

    for i in range(n):
        wait_time_i = completion_times[i][m - 1] - sum(times[sequence[i] - 1][j] for j in range(m))
        total_wait_time += wait_time_i

    return total_time, total_idle_time, total_wait_time

# вычисляет параметры Петрова (Pi1, Pi2, λ) и разбивает на подмножества D1, D0, D2
def calculate_petrov_parameters(times):
    n = len(times)
    m = len(times[0])

    # делит станки на две группы
    parameters = []
    m_half = math.floor(m / 2)

    for i in range(n):
        pi1 = sum(times[i][j] for j in range(m_half))
        pi2 = sum(times[i][j] for j in range(m_half, m))
        lambda_i = pi2 - pi1
        parameters.append({'part': i + 1, 'p1': pi1, 'p2': pi2, 'lambda': lambda_i})

    # сортирует детали по λ и создаёт три подмножества
    d1 = sorted([p for p in parameters if p['lambda'] > 0], key=lambda x: x['lambda'])
    d0 = sorted([p for p in parameters if p['lambda'] == 0], key=lambda x: x['lambda'])
    d2 = sorted([p for p in parameters if p['lambda'] < 0], key=lambda x: x['lambda'], reverse=True)

    return d1, d0, d2, parameters

# строит две последовательности по правилам Петрова
def generate_petrov_sequences(d1, d0, d2):
    sequence1 = [p['part'] for p in d1] + [p['part'] for p in d0] + [p['part'] for p in d2]
    sequence2 = [p['part'] for p in d2] + [p['part'] for p in d0] + [p['part'] for p in d1]
    return [sequence1, sequence2]

# создаёт случайную последовательность деталей
def generate_random_sequence(n):
    parts = list(range(1, n + 1))
    random.shuffle(parts)
    return parts


def brute_force(times):
    n = len(times)
    all_parts = list(range(1, n + 1))
    all_permutations = list(permutations(all_parts))

    best_sequence = None
    min_time = float('inf')

    for sequence in all_permutations:
        current_time, _, _ = calculate_total_time(list(sequence), times)
        if current_time < min_time:
            min_time = current_time
            best_sequence = list(sequence)

    return best_sequence, min_time


def main():
    processing_times = [
        [3, 2, 11, 12, 6, 2, 8],
        [4, 12, 3, 7, 7, 2, 2],
        [4, 8, 0, 12, 9, 1, 9],
        [13, 9, 2, 6, 1, 0, 2],
        [1, 14, 3, 4, 5, 6, 7],
        [16, 6, 3, 1, 2, 10, 1],
        [2, 10, 4, 0, 10, 11, 8]
    ]

    n = len(processing_times)

    print("Исходные данные:")
    for i, row in enumerate(processing_times):
        print(f"Деталь {i + 1}: {row}")

    d1, d0, d2, petrov_params = calculate_petrov_parameters(processing_times)

    print("\n" + "=" * 50 + "\n")
    print("Расчеты параметров Петрова:")
    for p in petrov_params:
        print(f"Деталь {p['part']}: Pi1 = {p['p1']}, Pi2 = {p['p2']}, λi =  = {p['lambda']}")
    print(f"\nПодмножества:")
    print(f"D1 (λ > 0): {[p['part'] for p in d1]}")
    print(f"D0 (λ = 0): {[p['part'] for p in d0]}")
    print(f"D2 (λ < 0): {[p['part'] for p in d2]}")

    print("\n" + "-" * 50 + "\n")
    print("Последовательности, полученные по правилам Петрова:")
    petrov_sequences = generate_petrov_sequences(d1, d0, d2)
    for i, seq in enumerate(petrov_sequences):
        t_mn, _, _ = calculate_total_time(seq, processing_times)
        print(f"Правило {i + 1}: Последовательность {seq}, Итоговое время: = {t_mn}")

    print("\n" + "-" * 50 + "\n")
    print("Случайная последовательность:")
    random_sequence = generate_random_sequence(n)
    t_mn_rand, _, _ = calculate_total_time(random_sequence, processing_times)
    print(f"Последовательность: {random_sequence}, Итоговое время: = {t_mn_rand}")

    print("\n" + "-" * 50 + "\n")
    print("Оптимальная последовательность (метод полного перебора):")
    best_sequence, min_time = brute_force(processing_times)
    print(f"Последовательность: {best_sequence}, Итоговое время: {min_time}")

if __name__ == "__main__":
    main()
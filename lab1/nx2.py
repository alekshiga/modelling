import matplotlib.pyplot as plt
import pandas as pd

def johnson(times):
    time_A = sorted([time for time in times if times[0] <= times[1]], key=lambda  x: x[0])
    time_B = sorted([time for time in times if times[0] > times[1]], key = lambda x: x[1], reverse=True)

    optimal_time = time_A + time_B
    return optimal_time

def()
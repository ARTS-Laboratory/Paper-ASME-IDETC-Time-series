# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:24:29 2023

@author: localuser
"""
import numpy as np
import matplotlib.pyplot as plt

num_samples = 500000
file_path = r'C:\Users\localuser\Downloads\inputData1_raw.txt'

def read_data_from_file(file_path):
    times, data = [], []
    with open(file_path, 'r') as file:
        for line in file:
            time, value = line.strip().split('\t')
            times.append(float(time))
            data.append(float(value))
    return times, data

def main():
    cusum = 0
    cusum_prev = 0
    shock = 0
    begin = 0
    n = 1
    avg = 0
    times, data = read_data_from_file(file_path)
    shock_intervals = []
    non_shock_intervals = []
    # Plot for the second line
    for i in range(num_samples):
        avg = ((avg * (n - 1)) + np.abs(data[i])) / n
        cusum += np.abs(data[i]) - avg
        n += 1
        if (cusum > cusum_prev+1.5) and shock==0:
            non_shock_intervals.append((times[begin], times[i-1]))
            begin = i
            shock = 1
        elif (cusum < cusum_prev+1.5) and shock==1:
            shock_intervals.append((times[begin], times[i-1]))
            begin = i
            shock = 0
        cusum_prev = cusum
    if (shock):
        shock_intervals.append((times[begin], times[times.__len__()-1]))
    else:
        non_shock_intervals.append((times[begin], times[times.__len__()-1]))
    fig, ax = plt.subplots()
    
    ax.plot(times, data, color='black')
    for start, end in shock_intervals:
        ax.axvspan(start, end, facecolor='green', alpha=0.3)
    for start, end in non_shock_intervals:
        ax.axvspan(start, end, facecolor='red', alpha=0.3)
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('Acceleration(m/s/s)')
    ax.set_title('Forced Vibration And Shock (Green=Shock, Red=Non-shock)')
    plt.show()

if __name__ == "__main__":
    main()

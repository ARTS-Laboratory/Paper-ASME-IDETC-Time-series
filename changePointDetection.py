# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:24:29 2023

@author: localuser
"""

import numpy as np
import matplotlib.pyplot as plt

num_samples = 500000
interval_size = 2000
file_path = r'C:\Users\localuser\Downloads\inputData1_raw.txt'
stddev_shock = 10
num_intervals = int(num_samples / interval_size)

def read_data_from_file(file_path):
    times, data = [], []
    with open(file_path, 'r') as file:
        for line in file:
            time, value = line.strip().split('\t')
            times.append(float(time))
            data.append(float(value))
    return times, data

def find_change_interval(data):
    stddev = np.std(data)
    return stddev

def main():
    times, data = read_data_from_file(file_path)
    shock_intervals = []
    non_shock_intervals = []
    for i in range(num_intervals):
        low = interval_size*i
        high = (interval_size-1)*(i+1)+i
        interval_stddev = find_change_interval(data[low:high])
        if interval_stddev > stddev_shock:
            shock_intervals.append((times[low], times[high - 1]))
        else:
            non_shock_intervals.append((times[low], times[high - 1]))
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

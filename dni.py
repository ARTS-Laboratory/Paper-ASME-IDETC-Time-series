# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 22:15:54 2024

@author: localuser
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:24:29 2023

@author: localuser
"""
import numpy as np
import matplotlib.pyplot as plt

num_samples = 500000
file_path = r'C:\Users\localuser\Downloads\inputData1_raw.txt'
d_threshold = 3
i_threshold = 15

def read_data_from_file(file_path):
    times, data = [], []
    with open(file_path, 'r') as file:
        for line in file:
            time, value = line.strip().split('\t')
            times.append(float(time))
            data.append(float(value))
    return times, data

def main():
    begin = 0
    shock = 0
    times, data = read_data_from_file(file_path)
    shock_intervals = []
    non_shock_intervals = []
    for i in range(num_samples):
        if ((np.abs(data[i]-data[i-1])>d_threshold) or np.abs(data[i])>i_threshold) and shock==0:
            non_shock_intervals.append((times[begin], times[i-1]))
            begin = i
            shock = 1
        elif ((np.abs(data[i]-data[i-1])<d_threshold) and np.abs(data[i])<i_threshold) and shock==1:
            shock_intervals.append((times[begin], times[i-1]))
            begin = i
            shock = 0
    if shock==0:
        non_shock_intervals.append((times[begin], times[times.__len__()-1]))
    else:
        shock_intervals.append((times[begin], times[times.__len__()-1]))

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

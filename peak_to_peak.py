# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:32:16 2023

@author: goshorna
"""

import numpy as np
import matplotlib.pyplot as plt

def read_data_from_file(file_path):
    times, data = [], []
    with open(file_path, 'r') as file:
        for line in file:
            time, value = line.strip().split('\t')
            times.append(float(time))
            data.append(float(value))
    return times, data

def find_change_interval(data):
    if data[0] > data[1]:
        minimum = data[1]
        maximum = data[0]
    else:
        minimum = data[0]
        maximum = data[1]
    for datum in data:
        if maximum < datum:
            maximum = datum
        if minimum > datum:
            minimum = datum
    if maximum - minimum > 40:
        return True
    else:
        return False

def main():
    file_path = r'C:\Users\localuser\Downloads\inputData1_raw.txt'  # Replace with the path to your data file
    times, data = read_data_from_file(file_path)
    shock_intervals = []
    non_shock_intervals = []
    for i in range(500):
        low = 1000 * i
        high = 999 * (i + 1) + i
        if find_change_interval(data[low:high]):
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

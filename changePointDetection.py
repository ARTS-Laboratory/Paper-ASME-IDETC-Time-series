# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:00:52 2023

@author: localuser
"""

import numpy as np
import matplotlib.pyplot as plt

num_samples = 500000
interval_size = 1000
file_path = r'C:\Users\localuser\Downloads\inputData1_raw.txt'
stddev_shock = 10
num_intervals = int(num_samples/interval_size)

def read_data_from_file(file_path):
    times, data = [], []
    with open(file_path, 'r') as file:
        for line in file:
            time, value = line.strip().split('\t')
            times.append(float(time))
            data.append(float(value))
    return times, data

def find_change_interval(data):
    avg = 0
    count = 0
    for datum in data:
        avg += datum
        count += 1
    avg = avg / count
    stddev = 0
    for datum in data:
        stddev += (datum - avg) * (datum - avg)
    stddev = stddev / count
    stddev = np.sqrt(stddev)
    return stddev
    

def exists_in(listIn, item):
    for obj in listIn:
        if obj == item:
            return True
    return False

def main():
    times, data = read_data_from_file(file_path)
    shock_intervals = []
    shock_data = []
    for i in range(num_intervals):
        low = interval_size*i
        high = (interval_size-1)*(i+1) + i
        if (find_change_interval(data[low : high])>stddev_shock):
            shock_intervals.append(times[low : high])
            shock_data.append(data[low: high])
    shock_intervals_flat = np.concatenate(shock_intervals)
    shock_data_flat = np.concatenate(shock_data)
    plt.plot(times, data, color='black')
    plt.scatter(shock_intervals_flat, shock_data_flat, color='blue')
    plt.xlabel('Time(s)')
    plt.ylabel('Acceleration(m/s/s)')
    plt.title('Forced Vibration And Shock(Blue=Shock prediction)')
    plt.show()
    
if __name__ == "__main__":
    main()

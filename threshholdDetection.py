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

def exists_in(listIn, item):
    for obj in listIn:
        if obj == item:
            return True
    return False

def main():
    file_path = r'C:\Users\localuser\Downloads\inputData1_raw.txt'  # Replace with the path to your data file
    times, data = read_data_from_file(file_path)
    shock_intervals = []
    shock_data = []
    for i in range(500):
        low = 0+(1000*i)
        high = 999*(i+1)
        if (find_change_interval(data[low : high])):
            shock_intervals.append(times[low : high])
            shock_data.append(data[low: high])
    shock_intervals_flat = np.concatenate(shock_intervals)
    shock_data_flat = np.concatenate(shock_data)
    fig, ax = plt.subplots()
    ax.plot(times, data, color='black')
    ax.scatter(shock_intervals_flat, shock_data_flat, color='blue')
    
if __name__ == "__main__":
    main()

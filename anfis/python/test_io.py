import csv
import numpy as np
x = []
y = []
with open('day6_10.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        int_row = [int(x) for x in row]
        x.append(int_row[:-1])
        y.append(int_row[-1])
print(np.shape(x), np.shape(y))
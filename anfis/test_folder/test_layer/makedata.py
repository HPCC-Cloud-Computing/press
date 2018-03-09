# Nhap cac thu vien
import numpy as np
from math import ceil, floor
import csv
import argparse
from utils import get_data

# Cac hang so va gia tri mac dinh
DEFAULT_OUTPUT_FILE_DIRECTORY = "output.csv"
DEFAULT_WINDOW_SIZE = 20
output_file_name = "w20.csv"

x = get_data(6, 10)


def convert_to_set(time_series, window_size):
    col_number = window_size + 1
    row_number = len(time_series) - window_size
    data_matrix = np.zeros((row_number, col_number), dtype=int)
    for i in np.arange(row_number):
        for j in np.arange(col_number):
            data_matrix[i][j] = int(time_series[j + i])
    return data_matrix


z = convert_to_set(time_series=x, window_size=DEFAULT_WINDOW_SIZE)

print(np.shape(z))
with open(output_file_name, "w+") as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(z)

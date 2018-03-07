# Nhap cac thu vien
import csv
from math import ceil, floor

import click
import numpy as np

from utils import get_data

# Cac hang so va gia tri mac dinh
DEFAULT_INPUT_FILE_DIRECTORY = "input.txt"
DEFAULT_OUTPUT_FILE_DIRECTORY = "output.csv"
DEFAULT_REPORT_FILE_DIRECTORY = "report.txt"
DEFAULT_MAX_BOX_REQUEST = 100
DEFAULT_VECTOR_SIZE = 10
DEFAULT_MODEL = 1
DEFAULT_SCALE_IN_MAX = 5
DEFAULT_SCALE_OUT_MAX = 5


# --------Mo ta chuc nang cua module pre_process---------------------
# -------------------------------------------------------------------
# -Input : Mot file chua chuoi cac gia tri so request trong tung ngay
# -Output: Mot file chua cac bo Vector, label tuong ung


# Ham tinh gia tri dai dien cua vector nham tinh delta
def represent_vector_value(vector, bms=DEFAULT_MAX_BOX_REQUEST):
    value = ceil(float(vector[-1] - vector[-2]) / float(bms))
    return value


# Ham xuat ra cac bo Vector, nhan
def request_to_sets(input_request_list, box_max_request=DEFAULT_MAX_BOX_REQUEST, vector_size=DEFAULT_VECTOR_SIZE,
                    scale_in_max=DEFAULT_SCALE_IN_MAX, scale_out_max=DEFAULT_SCALE_OUT_MAX):
    request_list = input_request_list
    if len(request_list) > vector_size:
        col_number = vector_size + 1
        row_number = len(request_list) - vector_size
        data_matrix = np.zeros((row_number, col_number), dtype=int)
        for i in np.arange(row_number):
            for j in np.arange(col_number - 1):
                data_matrix[i][j] = int(request_list[j + i])
            theta = represent_vector_value(request_list[i:i + vector_size - 1], box_max_request)
            if theta > 0:
                delta = ceil(theta)
            else:
                delta = floor(theta)
            if -scale_in_max <= theta <= scale_out_max:
                data_matrix[i][-1] = int(delta)
            elif delta < -scale_in_max:
                data_matrix[i][-1] = int(-scale_in_max)
            else:
                data_matrix[i][-1] = int(scale_out_max)
        return data_matrix
    else:
        print("Invalid vector_size. It can be greater than size of sequence")
        return None


@click.command()
@click.option("-o", "--output", default=DEFAULT_INPUT_FILE_DIRECTORY, help="Path to output file. Default value = "
                                                                           + str(DEFAULT_INPUT_FILE_DIRECTORY))
@click.option("--bmr", default=DEFAULT_MAX_BOX_REQUEST, help="Box max request. Default value = "
                                                             + str(DEFAULT_MAX_BOX_REQUEST))
@click.option("--vs", default=DEFAULT_VECTOR_SIZE, help="Vector size. Default value = "
                                                        + str(DEFAULT_VECTOR_SIZE))
@click.option("--sim", default=DEFAULT_SCALE_IN_MAX, help="Scale in max. Default value = "
                                                          + str(DEFAULT_SCALE_IN_MAX))
@click.option("--som", default=DEFAULT_SCALE_OUT_MAX, help="Scale out max. Default value = "
                                                           + str(DEFAULT_SCALE_OUT_MAX))
@click.option("--start", help="Start day for World Cup dataset")
@click.option("--end", help="End day for World Cup dataset")
def exec_arg(output, bmr: int, vs: int, sim: int, som: int, start: int, end: int):
    # Khai bao tham so dong lenh chuyen vao
    click.echo("Reading files...")
    request_list = get_data(int(start), int(end))
    click.echo("Success!! ...")
    click.echo('Processing ...')
    t = request_to_sets(request_list, box_max_request=bmr, vector_size=vs,
                        scale_in_max=sim, scale_out_max=som)
    print("Success !!")
    print("Writing output file " + str(output) + " ...")
    with open(output, "w+") as my_csv:
        csvwriter = csv.writer(my_csv, delimiter=',')
        csvwriter.writerows(t)
    print("Success !!")

    print("Finish! You can check output csv file.")


if __name__ == '__main__':
    exec_arg()

# Nhap cac thu vien
import numpy as np
from math import ceil, floor
import csv
import argparse
from utils import get_single_data, get_data

# Cac hang so va gia tri mac dinh
DEFAULT_INPUT_FILE_DIRECTORY = "input.txt"
DEFAULT_OUTPUT_FILE_DIRECTORY = "output.csv"
DEFAULT_REPORT_FILE_DIRECTORY = "report.txt"
DEFAULT_MAX_BOX_REQUEST = 30
DEFAULT_VECTOR_SIZE = 10
DEFAULT_MODEL = 1
DEFAULT_SCALE_IN_MAX = 50
DEFAULT_SCALE_OUT_MAX = 50


# --------Mo ta chuc nang cua module pre_process---------------------
# -------------------------------------------------------------------
# -Input : Mot file chua chuoi cac gia tri so request trong tung ngay
# -Output: Mot file chua cac bo Vector, label tuong ung


# Ham tinh gia tri dai dien cua vector nham tinh delta
def represent_vector_value(vector, bms=DEFAULT_MAX_BOX_REQUEST):
    value = ceil(float(vector[-1] - vector[-2]) / float(bms))
    return value


# Ham xuat ra cac bo Vector, nhan
def request_to_sets(input_request_list, box_max_request=DEFAULT_MAX_BOX_REQUEST,
                    vector_size=DEFAULT_VECTOR_SIZE,
                    scale_in_max=DEFAULT_SCALE_IN_MAX,
                    scale_out_max=DEFAULT_SCALE_OUT_MAX):
    request_list = input_request_list
    if len(request_list) > vector_size:
        col_number = vector_size + 1
        row_number = len(request_list) - vector_size
        data_matrix = np.zeros((row_number, col_number), dtype=int)
        for i in np.arange(row_number):
            for j in np.arange(col_number - 1):
                data_matrix[i][j] = int(request_list[j + i])
            theta = represent_vector_value(request_list[i:i + vector_size - 1],
                                           box_max_request)
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


if __name__ == '__main__':

    # Khai bao tham so dong lenh chuyen vao
    parser = argparse.ArgumentParser()

    # Input file directory
    parser.add_argument("-o", "--output",
                        help="Path to output file. Default value = "
                             + str(DEFAULT_INPUT_FILE_DIRECTORY))

    # Output file directory
    parser.add_argument("-i", "--input",
                        help="Path to input file. Default value = "
                             + str(DEFAULT_OUTPUT_FILE_DIRECTORY))

    # Box max request
    parser.add_argument("--bmr", help="Box max request. Default value = "
                                      + str(DEFAULT_MAX_BOX_REQUEST))

    # Vector size
    parser.add_argument("--vs", help="Vector size. Default value = "
                                     + str(DEFAULT_VECTOR_SIZE))

    # Scale in max
    parser.add_argument("--sim", help="Scale in max. Default value = "
                                      + str(DEFAULT_SCALE_IN_MAX))

    # Scale out max
    parser.add_argument("--som", help="Scale out max. Default value = "
                                      + str(DEFAULT_SCALE_OUT_MAX))

    # Start day dataset
    parser.add_argument("--start",
                        help="Start day for World Cup dataset")

    # End day dataset
    parser.add_argument("--end",
                        help="End day for World Cup dataset")

    args = vars(parser.parse_args())
    print("Reading files ...")

    # Kiem tra voi tung tham so dong lenh
    if args.get("input", None) is None:
        if args.get("start", None) is None and args.get("end", None) is None:
            input_file_name = DEFAULT_INPUT_FILE_DIRECTORY
            request_list = get_single_data(input_file_name)
        elif int(args["end"]) >= int(args["start"]):
            request_list = get_data(int(args["start"]), int(args["end"]))
        else:
            print("Error! Start day can't be greater than end day")
    elif args.get("start", None) is None and args.get("end", None) is None:
        input_file_name = args["input"]
        request_list = get_single_data(input_file_name)
    else:
        print(
            "Syntax Error! Please check command again or type python preprocess -h to get more information")
        exit(1)
    print("Success !!")
    print("Processing ...")
    if args.get("output", None) is None:
        output_file_name = DEFAULT_OUTPUT_FILE_DIRECTORY
    else:
        output_file_name = args["output"]

    if args.get("bmr", None) is None:
        bmr = DEFAULT_MAX_BOX_REQUEST
    else:
        bmr = args["bmr"]

    if args.get("vs", None) is None:
        vs = DEFAULT_VECTOR_SIZE
    else:
        vs = args["vs"]

    if args.get("sim", None) is None:
        sim = DEFAULT_SCALE_IN_MAX
    else:
        sim = args["sim"]

    if args.get("som", None) is None:
        som = DEFAULT_SCALE_OUT_MAX
    else:
        som = args["som"]
    t = request_to_sets(request_list, box_max_request=bmr, vector_size=vs,
                        scale_in_max=sim, scale_out_max=som)
    print("Success !!")
    print("Writing output file " + str(output_file_name) + " ...")
    with open(output_file_name, "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(t)
    print("Success !!")

    print("Finish! You can check output csv file.")

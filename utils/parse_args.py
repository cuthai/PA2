import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lf', '--load_file', type=str, default='updated_iris.csv',
                        help='Specify file to load from input file')
    parser.add_argument('-p1', '--part_1', action='store_true',
                        help='Part 1 in PDF: Perform Data Cleansing on the data and output to output folder')
    parser.add_argument('-p2', '--part_2', action='store_true',
                        help='Part 2 in PDF: Perform Feature Generation on the data and output to output folder')

    args = parser.parse_args()

    return args

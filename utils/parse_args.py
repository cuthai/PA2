import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lf', '--load_file', type=str, default='updated_iris.csv',
                        help='Specify file to load from input file')
    parser.add_argument('-of', '--output_files', action='store_true',
                        help='Specify output for individual parts in output folder')

    args = parser.parse_args()

    return args

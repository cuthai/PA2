import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lf', '--load_file', type=str, default='iris_data_for_cleansing.csv',
                        help='Specify file to load from input file')

    args = parser.parse_args()

    return args

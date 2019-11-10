from utils import parse_args
from preprocessing.load_data import load_data
from preprocessing.data_cleansing import data_cleansing


def main():
    args = parse_args.parse_args()

    df = load_data(args.load_file)

    df = data_cleansing(df)


if __name__ == '__main__':
    main()
from utils import parse_args
from preprocessing.load_data import load_data
from preprocessing.data_cleansing import data_cleansing
from preprocessing.feature_generation import feature_generation_pca


def main():
    args = parse_args.parse_args()

    df = load_data(args.load_file)

    if args.part_1:
        df = data_cleansing(df)

    if args.part_2:
        feature_generation_pca(df, 2)


if __name__ == '__main__':
    main()
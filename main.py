from utils import parse_args
from preprocessing.load_data import load_data
from preprocessing.data_cleansing import data_cleansing_class_mean
from preprocessing.feature_generation import feature_generation_pca
from preprocessing.feature_preprocessing import feature_preprocessing_confidence_interval


def main():
    args = parse_args.parse_args()

    df = load_data(args.load_file)

    if args.part_1:
        df = data_cleansing_class_mean(df)

    if args.part_2:
        df = feature_generation_pca(df, 2)

    if args.part_3:
        feature_preprocessing_confidence_interval(df)


if __name__ == '__main__':
    main()
from utils import parse_args
from preprocessing.load_data import load_data
from preprocessing.data_cleansing import data_cleansing_class_mean
from preprocessing.feature_generation import feature_generation_pca
from preprocessing.feature_preprocessing import feature_preprocessing_confidence_interval
from preprocessing.feature_ranking import feature_ranking_fisher_lda
from preprocessing.dimensionality_reduction import dimensionality_reduction_pca


def main():
    args = parse_args.parse_args()

    df = load_data(args.load_file)

    if args.part_1:
        data_cleansing_class_mean(df)

    if args.part_2:
        feature_generation_pca(df, 2)

    if args.part_3:
        df = feature_preprocessing_confidence_interval(df)

    if args.part_4:
        feature_ranking_fisher_lda(df)

    if args.part_5:
        df = dimensionality_reduction_pca(df, 2)


if __name__ == '__main__':
    main()
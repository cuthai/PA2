import json


def feature_ranking_fisher_lda(df):
    fisher_lda_results = calculate_fisher_lda(df)

    df = rank_and_select_features(df, fisher_lda_results)

    output_feature_ranking_result(df, fisher_lda_results)

    return df


def calculate_fisher_lda(df):
    class_mean = df.groupby(by='class').mean()
    class_standard_deviation = df.groupby(by='class').std()
    feature_fdr_dict = {}

    for feature_index in range(6):
        feature_fdr = 0

        for class_index in range(3):
            class_fdr = 0

            for non_class_index in range(3):
                if class_index != non_class_index:
                    sb = (class_mean.iat[class_index, feature_index] -
                          class_mean.iat[non_class_index, feature_index]) ** 2
                    sw = class_standard_deviation.iat[class_index, feature_index] + \
                         class_standard_deviation.iat[non_class_index, feature_index]
                    class_fdr += (sb / sw)

            feature_fdr += class_fdr

        feature_fdr_dict.update({df.keys()[feature_index]: feature_fdr})

    return feature_fdr_dict


def rank_and_select_features(df, fisher_lda_results):
    max_1_fdr_score = 0
    max_1_feature = None
    max_2_fdr_score = 0
    max_2_feature = None

    for feature, score in fisher_lda_results.items():
        if score > max_1_fdr_score:
            if max_1_fdr_score > max_2_fdr_score:
                max_2_fdr_score = max_1_fdr_score
                max_2_feature = max_1_feature

            max_1_fdr_score = score
            max_1_feature = feature

        elif score > max_2_fdr_score:
            max_2_fdr_score = score
            max_2_feature = feature

    df = df[[max_1_feature, max_2_feature]]

    return df


def output_feature_ranking_result(df, fisher_lda_results):
    df.to_csv("output/part4_feature_ranking_data.csv", index=False)

    with open("output/part4_feature_ranking_algorithm.json", 'w') as file:
        json.dump(fisher_lda_results, file)

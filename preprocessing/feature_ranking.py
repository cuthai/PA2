import json


def feature_ranking_fisher_lda(df):
    """
    Main method to rank features using Fisher's LDA

    :param df: Loaded data as a DataFrame
    :return df: DataFrame with the two highest ranking features
    """
    # Perform Fisher's LDA
    fisher_lda_results = calculate_fisher_lda(df)

    # Rank and select the features using the results from Fisher's LDA
    df = rank_and_select_features(df, fisher_lda_results)

    # Output data to a csv
    output_feature_ranking_result(df, fisher_lda_results)

    return df


def calculate_fisher_lda(df):
    """
    For the 6 features: the 4 original nd 2 new features, this method calculates a FDR score and returns the FDR scores

    :param df: Loaded data as a DataFrame
    :return feature_fdr_dict: Dictionary with column name and that column's FDR score
    """
    # Calculate test statistics
    class_mean = df.groupby(by='class').mean()
    class_standard_deviation = df.groupby(by='class').std()
    feature_fdr_dict = {}

    # Calculate FDR over each column (feature)
    for feature_index in range(6):
        feature_fdr = 0

        # Split data by class for that column
        for class_index in range(3):
            class_fdr = 0

            # Sum sb and sw and calculate fdr over the current class and non class data
            for non_class_index in range(3):
                if class_index != non_class_index:
                    sb = (class_mean.iat[class_index, feature_index] -
                          class_mean.iat[non_class_index, feature_index]) ** 2
                    sw = class_standard_deviation.iat[class_index, feature_index] + \
                         class_standard_deviation.iat[non_class_index, feature_index]
                    class_fdr += (sb / sw)

            # Sum fdr for the column by class
            feature_fdr += class_fdr

        # Update our dictionary
        feature_fdr_dict.update({df.keys()[feature_index]: feature_fdr})

    return feature_fdr_dict


def rank_and_select_features(df, fisher_lda_results):
    """
    Method takes FDR scores and returns a DataFrame with the two columns with the highest FDR scores

    :param df: Loaded data as a DataFrame
    :param fisher_lda_results: FDR scores by column
    :return df: Two highest scoring features
    """
    max_1_fdr_score = 0
    max_1_feature = None
    max_2_fdr_score = 0
    max_2_feature = None

    # Go through each item in the Fisher LDA results dictionary
    for feature, score in fisher_lda_results.items():
        # Assign highest score to max_1
        if score > max_1_fdr_score:

            # If the max_1 has changed, assign the old max_1 to max_2
            if max_1_fdr_score > max_2_fdr_score:
                max_2_fdr_score = max_1_fdr_score
                max_2_feature = max_1_feature

            max_1_fdr_score = score
            max_1_feature = feature

        # If max_1 check failed, try again for max_2
        elif score > max_2_fdr_score:
            max_2_fdr_score = score
            max_2_feature = feature

    # Set df to two highest features
    df = df[[max_1_feature, max_2_feature]]

    return df


def output_feature_ranking_result(df, fisher_lda_results):
    """
    Method for outputting current DataFrame and Fisher's LDA rankings at the end of this module to a csv

    :param df: Passed in DataFrame for this module
    :param fisher_lda_results: FDR scores by column
    """
    df.to_csv("output/part4_feature_ranking_data.csv")

    with open("output/part4_feature_ranking_algorithm.json", 'w') as file:
        json.dump(fisher_lda_results, file)

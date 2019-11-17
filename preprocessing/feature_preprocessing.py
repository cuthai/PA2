import json


def feature_preprocessing_confidence_interval(df):
    """
    Main method to preprocess data by removing outliers using z score normalization and confidence intervals

    :param df: Loaded data as a DataFrame
    :return df: df with removed outliers
    """
    normalize_df = df.copy(deep=True)

    # Normalize the data
    normalize_df = normalize_z_score(normalize_df)

    # Construct confidence intervals and tag values outside of those intervals
    outlier_set, confidence_interval = construct_confidence_interval_and_find_outliers(normalize_df, 1.96)

    # Remove values outside of those intervals
    df, outlier_df = remove_outliers(df, outlier_set)

    # Output data to a csv
    output_feature_preprocessing_result(df, outlier_df, confidence_interval)

    return df


def normalize_z_score(df):
    """
    Method to normalize all data based on their z scores. This method focuses on the four original features

    :param df: Copy of loaded data as a DataFrame
    :return df: Normalized df
    """
    # Z score normalization by column for the four original features
    for column in df.keys()[:-5]:

        # Get column test statistics
        mean = df[column].mean()
        standard_deviation = df[column].std()

        # Z score normalization
        for index in range(150):
            current_data = df.at[index, column]
            df.at[index, column] = (current_data - mean) / standard_deviation

    return df


def construct_confidence_interval_and_find_outliers(df, t_value):
    """
    Method for constructing confidence intervals for each column (of the four original)

    :param df: Normalized df
    :param t_value: Test statistic for constructing confidence intervals
    :return outlier_set: List of indices outside of confidence intervals
    :return confidence_interval: Dictionary containing the calculated confidence intervals by column
    """
    # Use a set for the outlier set since only distinct indices matter
    outlier_set = set()
    confidence_interval = {}

    # Calculate confidence intervals by column
    for column in df[["sepal length", "sepal width", "petal length", "petal width"]].keys():

        # Calculate test statistic, break df into two parts
        mean = df[column].mean()
        slower = df.loc[df[column] < mean][column].mean()
        supper = df.loc[df[column] > mean][column].mean()

        # Calculate upper and lower bounds
        lower_bound = slower - (t_value * (mean - slower))
        upper_bound = supper + (t_value * (supper - mean))

        # Add to dictionary
        temp_dict = {column: {'lower_bound': lower_bound, 'upper_bound': upper_bound}}
        confidence_interval.update(temp_dict)

        # Add any indices outside of the confidence interval to the outlier set
        for index in range(150):
            if df.at[index, column] < lower_bound:
                outlier_set.add(index)
            elif df.at[index, column] > upper_bound:
                outlier_set.add(index)

    return outlier_set, confidence_interval


def remove_outliers(df, outlier_set):
    """
    Method removes indices tagged as outliers from the DataFrame

    :param df: Loaded data as a DataFrame, not normalized
    :param outlier_set: Set with indices that were marked as outliers
    :return df: DataFrame with outlier removes
    :return removed_df: Dataframe with the outliers
    """
    # Create removed_df first with the outliers
    removed_df = df.loc[df.index.isin(outlier_set)]

    # Set df equal to non outliers
    df = df.loc[~df.index.isin(outlier_set)]

    return df, removed_df


def output_feature_preprocessing_result(df, outlier_df, confidence_interval):
    """
    Method for outputting current DataFrame and confidence intervals at the end of this module to a csv

    :param df: Passed in DataFrame for this module
    :param outlier_df: DataFrame with outliers
    :param confidence_interval: Confidence Intervals by column
    """
    df.to_csv("output/part3_feature_preprocessing_data.csv")
    outlier_df.to_csv("output/part3_feature_preprocessing_outlier_data.csv")

    with open("output/part3_feature_preprocessing_outlier_algorithm.json", 'w') as file:
        json.dump(confidence_interval, file)

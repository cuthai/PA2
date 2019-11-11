def feature_preprocessing_confidence_interval(df):
    normalize_df = df.copy(deep=True)

    normalize_df = normalize_z_score(normalize_df)

    outlier_set = construct_confidence_interval_and_find_outliers(normalize_df, 1.96)

    df, outlier_df = remove_outliers(df, outlier_set)

    output_feature_preprocessing_result(df, outlier_df)

    return df


def normalize_z_score(df):
    for column in df.keys()[:-5]:
        mean = df[column].mean()
        standard_deviation = df[column].std()

        for index in range(150):
            current_data = df.at[index, column]
            df.at[index, column] = (current_data - mean) / standard_deviation

    return df


def construct_confidence_interval_and_find_outliers(df, t_value):
    outlier_set = set()

    for column in df.keys()[:-5]:
        low_outlier_index_list = []
        high_outlier_index_list = []

        mean = df[column].mean()
        slower = df.loc[df[column] < mean][column].mean()
        supper = df.loc[df[column] > mean][column].mean()

        lower_bound = slower - (t_value * (mean - slower))
        upper_bound = supper + (t_value * (supper - mean))

        for index in range(150):
            if df.at[index, column] < lower_bound:
                outlier_set.add(index)
            elif df.at[index, column] > upper_bound:
                outlier_set.add(index)

    return outlier_set


def remove_outliers(df, outlier_set):
    removed_df = df.loc[df.index.isin(outlier_set)]
    df = df.loc[~df.index.isin(outlier_set)]

    return df, removed_df


def output_feature_preprocessing_result(df, outlier_df):
    df.to_csv("output/part3_feature_preprocessing_result.csv", index=False)
    outlier_df.to_csv("output/part3_feature_preprocessing_outlier_result.csv", index=False)

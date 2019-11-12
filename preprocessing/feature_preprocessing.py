import json


def feature_preprocessing_confidence_interval(df):
    normalize_df = df.copy(deep=True)

    normalize_df = normalize_z_score(normalize_df)

    outlier_set, confidence_interval = construct_confidence_interval_and_find_outliers(normalize_df, 1.96)

    df, outlier_df = remove_outliers(df, outlier_set)

    output_feature_preprocessing_result(df, outlier_df, confidence_interval)

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
    confidence_interval = {}

    for column in df[["sepal length", "sepal width", "petal length", "petal width"]].keys():
        mean = df[column].mean()
        slower = df.loc[df[column] < mean][column].mean()
        supper = df.loc[df[column] > mean][column].mean()

        lower_bound = slower - (t_value * (mean - slower))
        upper_bound = supper + (t_value * (supper - mean))

        temp_dict = {column: {'lower_bound': lower_bound, 'upper_bound': upper_bound}}
        confidence_interval.update(temp_dict)

        for index in range(150):
            if df.at[index, column] < lower_bound:
                outlier_set.add(index)
            elif df.at[index, column] > upper_bound:
                outlier_set.add(index)

    return outlier_set, confidence_interval


def remove_outliers(df, outlier_set):
    removed_df = df.loc[df.index.isin(outlier_set)]
    df = df.loc[~df.index.isin(outlier_set)]

    return df, removed_df


def output_feature_preprocessing_result(df, outlier_df, confidence_interval):
    df.to_csv("output/part3_feature_preprocessing_data.csv")
    outlier_df.to_csv("output/part3_feature_preprocessing_outlier_data.csv")

    with open("output/part3_feature_preprocessing_outlier_algorithm.json", 'w') as file:
        json.dump(confidence_interval, file)

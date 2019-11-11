import pandas as pd


def data_cleansing_class_mean(df):
    df = calculate_mean_and_fill(df)

    output_data_cleansing_result(df)

    return df


def calculate_mean_and_fill(df):
    cleansed_df = pd.DataFrame()

    for class_index in range(1, 4):
        class_df = df.loc[df["class"] == class_index]
        fill_nan_mapper = {}

        for column in class_df.keys()[:-1]:
            column_mean_value = class_df[column].mean()
            fill_nan_mapper.update({column: column_mean_value})

        cleansed_df = cleansed_df.append(class_df.fillna(fill_nan_mapper))

    return cleansed_df


def output_data_cleansing_result(df):
    df.to_csv("output/part1_data_cleansing_result.csv", index=False)

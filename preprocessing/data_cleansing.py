import pandas as pd


def data_cleansing(df):
    df = fill_nan_df(df)

    output_data_cleansing_result(df)

    return df


def fill_nan_df(df):
    cleansed_df = pd.DataFrame()

    for class_index in range(1, 4):
        temp_df = df.loc[df["class"] == class_index]
        fill_nan_mapper = {}

        for column in temp_df.keys()[:-1]:
            column_mean_value = temp_df[column].mean()
            fill_nan_mapper.update({column: column_mean_value})

        cleansed_df = cleansed_df.append(temp_df.fillna(fill_nan_mapper))

    return cleansed_df


def output_data_cleansing_result(df):
    df.to_csv("output/part1_data_cleansing_result.csv", index=False)

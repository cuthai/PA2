import pandas as pd


def data_cleansing_class_mean(df):
    """
    Main method of data_cleansing, runs code to calculate the mean value of each column by class,
        and places that value into null cells

    :param df: loaded data as a DataFrame
    :return df: Passed in DataFrame with filled in null values
    """
    # Perform null value replacement using mean value
    df = calculate_mean_and_fill(df)

    # Output data to a csv
    output_data_cleansing_result(df)

    return df


def calculate_mean_and_fill(df):
    """
    Method loops through each class, and calculates a dictionary that maps that class's columns to a mean value
        That dictionary is then used to fill in null cells

    :param df: loaded data as a DataFrame
    :return cleansed_df: Passed in DataFrame with filled in null values
    """
    cleansed_df = pd.DataFrame()

    # Loop through each class
    for class_index in range(1, 4):
        class_df = df.loc[df["class"] == class_index]
        fill_nan_mapper = {}

        # Narrow down to the column
        for column in class_df.keys()[:-1]:
            column_mean_value = class_df[column].mean()

            # Add to dictionary that maps class's columns to a mean value
            fill_nan_mapper.update({column: column_mean_value})

        # Update null values for that class and append that to the cleansed DataFrame
        cleansed_df = cleansed_df.append(class_df.fillna(fill_nan_mapper))

    return cleansed_df


def output_data_cleansing_result(df):
    """
    Method for outputting current DataFrame at the end of this module to a csv

    :param df: Passed in DataFrame for this module
    """
    df.to_csv("output/part1_data_cleansing_data.csv")

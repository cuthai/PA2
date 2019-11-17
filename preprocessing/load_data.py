import pandas as pd


def load_data(file_to_load_name):
    """
    Method for loading the data from a csv into a DataFrame

    :param file_to_load_name: Name of file to be loaded
    :return df: loaded data as a DataFrame
    """
    file_to_load_location = f'input/{file_to_load_name}'
    df = pd.read_csv(file_to_load_location)

    return df

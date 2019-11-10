import pandas as pd


def load_data(file_to_load_name):
    file_to_load_location = f'input/{file_to_load_name}'
    df = pd.read_csv(file_to_load_location)

    return df

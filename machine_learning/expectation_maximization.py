import random
import math
import numpy as np


def expectation_maximization(df):
    em_df = df.copy(deep=True)

    choose_random_starting_clusters(em_df)

    mean, standard_deviation, probabilities = calculate_parameters(em_df)

    calculate_e_step(em_df, mean, standard_deviation, probabilities)


def choose_random_starting_clusters(df):
    index = df.index.tolist()

    random.shuffle(index)

    first_split = (math.ceil(len(index) / 3))
    second_split = (math.ceil((len(index) / 3) * 2))

    first_part = index[0:first_split]
    second_part = index[first_split:second_split]
    third_part = index[second_split:len(index)]

    df['cluster'] = 0
    df.at[first_part, 'cluster'] = 1
    df.at[second_part, 'cluster'] = 2
    df.at[third_part, 'cluster'] = 3


def calculate_parameters(df):
    mean = df.groupby('cluster').mean()
    standard_deviation = df.groupby('cluster').std()
    probabilities = df.groupby('cluster').count() / len(df)

    return mean, standard_deviation, probabilities


def calculate_e_step(df, mean, standard_deviation, probabilities):
    n = len(df)

    for cluster in range(1, 4):
        data_array = df[["PCA 1", "PCA 2"]].values
        mean_array = mean.loc[mean.index == cluster].values
        covariance_array = np.cov(data_array.T)
        probabilities_array = probabilities.loc[mean.index == cluster].values

        normalizer = (1 / ((math.sqrt(2 * math.pi) * covariance_array) ** n))
        exponent = -.5 * ((data_array - mean_array).dot(np.linalg.inv(covariance_array)) ** 2)
        gaussian = normalizer.dot(np.exp(exponent).T)
        pass

import random
import math
import numpy as np


def expectation_maximization(df):
    em_df = df.copy(deep=True)

    choose_random_starting_clusters(em_df)

    data_array, mean, covariance, probabilities = calculate_initial_parameters(em_df)

    mixture = calculate_e_step(data_array, mean, covariance, probabilities)

    for i in range(100):
        new_mean, new_covariance, new_probabilities = calculate_m_step(data_array, mixture)

        test = 0
        for index in range(3):
            test += abs(new_mean[index].sum() - mean[index].sum())

        if test <= .00000001:
            break
        else:
            mean = new_mean
            covariance = new_covariance
            probabilities = new_probabilities
            mixture = calculate_e_step(data_array, mean, covariance, probabilities)
            print(covariance)

    pass


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


def calculate_initial_parameters(df):
    data_array = df[[["PCA 1", "PCA 2"]]].values
    mean = []
    covariance = []
    probabilities = []

    for index in range(1, 4):
        temp_df = df.loc[df['cluster'] == index][["PCA 1", "PCA 2"]]
        mean.append(temp_df.mean().values)
        covariance.append(np.cov(temp_df.values.T))
        probabilities.append(len(temp_df) / len(df))

    return data_array, mean, covariance, probabilities


def calculate_e_step(data_array, mean, covariance, probabilities):
    gaussian = []

    for index in range(3):
        mean_array = mean[index]
        covariance_array = covariance[index]
        probability = probabilities[index]

        normalizer_step1 = ((2 * math.pi) ** 2)
        normalizer_step2 = np.linalg.det(covariance_array)
        normalizer = 1 / math.sqrt(normalizer_step1 * normalizer_step2)

        exponent_step1 = data_array - mean_array
        exponent_step2 = -.5 * exponent_step1
        exponent_step3 = exponent_step2.dot(np.linalg.inv(covariance_array))
        exponent_step4 = exponent_step3 * exponent_step1
        exponent = np.exp(exponent_step4)

        gaussian.append(probability * (np.exp(exponent) * normalizer))

    mixture = []
    denominator = gaussian[0] + gaussian[1] + gaussian[2]
    for index in range(3):
        mixture.append(gaussian[index] / denominator)

    return mixture


def calculate_m_step(data_array, mixture):
    mean = []
    covariance = []
    probabilities = []
    x = []

    for index in range(3):
        mixture_sum = mixture[index].sum(axis=0)
        x.append(mixture_sum)

        mean_step1 = (mixture[index] * data_array).sum(axis=0)
        new_mean = mean_step1 / mixture_sum
        mean.append(new_mean)

        covariance_step1 = data_array - new_mean
        covariance_step2 = (mixture[index] * covariance_step1).T.dot(covariance_step1)
        new_covariance = covariance_step2 / mixture_sum
        covariance.append(new_covariance)

        probabilities.append(mixture_sum / len(data_array))

    return mean, covariance, probabilities

import random
import math
import numpy as np
from scipy.stats import multivariate_normal


def expectation_maximization(df, pca_df):
    em_df = pca_df.copy(deep=True)
    df = df.copy(deep=True)

    choose_random_starting_clusters(em_df)

    data_array, mean, covariance = calculate_initial_parameters(em_df)

    mixture = calculate_e_step(data_array, mean, covariance)

    mixture = iterate_expectation_maximization(data_array, mixture, mean)

    em_result_df = classify(df, em_df, mixture)

    output_expectation_maximization_result(em_result_df)


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
    data_array = df[["PCA 1", "PCA 2"]].values
    mean = []
    covariance = []

    for index in range(1, 4):
        temp_df = df.loc[df['cluster'] == index][["PCA 1", "PCA 2"]]
        mean.append(temp_df.mean().values)
        covariance.append(np.cov(temp_df.values.T))

    return data_array, mean, covariance


def calculate_e_step(data_array, mean, covariance):
    gaussian = []

    for index in range(3):
        gaussian_model = multivariate_normal(mean=mean[index], cov=covariance[index])
        gaussian.append(gaussian_model.pdf(data_array))

    mixture = []
    denominator = gaussian[0] + gaussian[1] + gaussian[2]
    for index in range(3):
        mixture_array = (gaussian[index] / denominator)
        mixture.append(mixture_array.reshape(len(data_array), 1))

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


def iterate_expectation_maximization(data_array, mixture, mean):
    for index in range(1000):
        new_mean, new_covariance, new_probabilities = calculate_m_step(data_array, mixture)

        test = 0
        for class_index in range(3):
            test += abs(new_mean[class_index].sum() - mean[class_index].sum())

        if test <= .00000001:
            break
        else:
            mean = new_mean
            covariance = new_covariance
            mixture = calculate_e_step(data_array, mean, covariance)

    return mixture


def classify(df, em_df, mixture):
    classification = []
    df = df.merge(em_df[["PCA 1", "PCA 2"]], left_index=True, right_index=True)

    for index in range(len(df)):
        if mixture[0][index] > mixture[1][index] and mixture[0][index] > mixture[2][index]:
            classification.append(1)
        elif mixture[1][index] > mixture[0][index] and mixture[1][index] > mixture[2][index]:
            classification.append(2)
        elif mixture[2][index] > mixture[0][index] and mixture[2][index] > mixture[1][index]:
            classification.append(3)

    df['em_class'] = classification

    return df


def output_expectation_maximization_result(df):
    df.to_csv("output/part6a_expectation_maximization_data.csv")

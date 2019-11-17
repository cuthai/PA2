import random
import math
import numpy as np
from scipy.stats import multivariate_normal


def expectation_maximization(df, pca_df):
    """
    Main method for classification using Expectation Maximization
        Uses the PCA features from part 5

    :param df: Loaded data as a DataFrame
    :param pca_df: DataFrame with PCA generated features
    """
    # Create copies to avoid modifying the original DataFrames
    em_df = pca_df.copy(deep=True)
    df = df.copy(deep=True)

    # Choose random starting clusters for initial statistics
    choose_random_starting_clusters(em_df)

    # Calculate initial parameters from the randomly selected clusters
    data_array, mean, covariance = calculate_initial_parameters(em_df)

    # Call to the iteration and get the results, iteration handles the e and m stages
    results = iterate_expectation_maximization(data_array, mean, covariance)

    # Add classification results to df
    em_result_df = classify(df, em_df, results)

    # Output results to a CSV
    output_expectation_maximization_result(em_result_df)


def choose_random_starting_clusters(df):
    """
    Method for choosing random clusters

    :param df: DataFrame with PCA generated features
    :return df: DataFrame with PCA generated features and randomly assigned starting classes
    """
    # Create a list of the indices
    index = df.index.tolist()

    # Randomly shuffle the list
    random.shuffle(index)

    # Calculate splits
    first_split = (math.ceil(len(index) / 3))
    second_split = (math.ceil((len(index) / 3) * 2))

    # Break index list based on splits
    first_part = index[0:first_split]
    second_part = index[first_split:second_split]
    third_part = index[second_split:len(index)]

    # Assign clusters based on broken up index list
    df['cluster'] = 0
    df.at[first_part, 'cluster'] = 1
    df.at[second_part, 'cluster'] = 2
    df.at[third_part, 'cluster'] = 3


def calculate_initial_parameters(df):
    """
    Method for calculating initial parameters for the randomly chosen clusters

    :param df: DataFrame with PCA generated features
    :return data_array: df as a numpy array
    :return mean: List of means by randomly chosen clusters
    :return covariance: List of covariance matrices by randomly chosen clusters
    """
    data_array = df[["PCA 1", "PCA 2"]].values
    mean = []
    covariance = []

    # Calculate test statistics by cluster
    for index in range(1, 4):
        temp_df = df.loc[df['cluster'] == index][["PCA 1", "PCA 2"]]
        mean.append(temp_df.mean().values)
        covariance.append(np.cov(temp_df.values.T))

    return data_array, mean, covariance


def iterate_expectation_maximization(data_array, mean, covariance):
    """
    Method for handling iteration through e and m steps

    :param data_array: numpy array of data
    :param mean: List of means by initial clusters
    :param covariance: List of covariance matrices by initial clusters
    :return mixture: Ending mixture probabilities for each row and by cluster
    """
    # Initial mixture calculation
    mixture = calculate_e_step(data_array, mean, covariance)

    # Maximum range is 1000
    for index in range(1000):
        # M-step
        new_mean, new_covariance, new_probabilities = calculate_m_step(data_array, mixture)

        # Create test statistic based off of changed means for convergence test
        test = 0
        for class_index in range(3):
            test += abs(new_mean[class_index].sum() - mean[class_index].sum())

        # Test convergence
        if test <= .00000001:
            # Break from for loop if converged
            break

        # Else set test statistics = new test statistics and perform E-step
        else:
            mean = new_mean
            covariance = new_covariance
            mixture = calculate_e_step(data_array, mean, covariance)

    return mixture


def calculate_e_step(data_array, mean, covariance):
    """
    Method for handling E-step, calculation of mixture probabilities based on current cluster test-statistics
        Uses Gaussian Model from scipy to create a multivariate PDF

    :param data_array: numpy array of data
    :param mean: List of means by current clusters
    :param covariance: List of covariance matrices by current clusters
    :return mixture: Current mixture probabilities for each row and by cluster
    """
    gaussian = []

    # Generate gaussian PDFs for the current clusters
    for index in range(3):
        gaussian_model = multivariate_normal(mean=mean[index], cov=covariance[index])
        gaussian.append(gaussian_model.pdf(data_array))

    # Calculate mixtures for each row by class
    mixture = []
    denominator = gaussian[0] + gaussian[1] + gaussian[2]
    for index in range(3):
        mixture_array = (gaussian[index] / denominator)
        mixture.append(mixture_array.reshape(len(data_array), 1))

    return mixture


def calculate_m_step(data_array, mixture):
    """
    Method for handling M-step, recalculation of test-statistics based on current mixtures

    :param data_array: numpy array of data
    :param mixture: Current mixture probabilities for each row and by cluster
    :return mean: List of means by current mixture
    :return covariance: List of covariance matrices by current mixture
    :return probabilities: List of probabilities by current mixture
    """
    mean = []
    covariance = []
    probabilities = []

    # Calculate new test statistics over each class's current mixtures
    for index in range(3):
        # Sum mixtures
        mixture_sum = mixture[index].sum(axis=0)

        # Mean calculation
        mean_step1 = (mixture[index] * data_array).sum(axis=0)
        new_mean = mean_step1 / mixture_sum
        mean.append(new_mean)

        # Covariance calculation
        covariance_step1 = data_array - new_mean
        covariance_step2 = (mixture[index] * covariance_step1).T.dot(covariance_step1)
        new_covariance = covariance_step2 / mixture_sum
        covariance.append(new_covariance)

        # Probabilities calculation
        probabilities.append(mixture_sum / len(data_array))

    return mean, covariance, probabilities


def classify(df, em_df, mixture):
    """
    Method for classifying data based on final mixtures

    :param df: Loaded data as a DataFrame
    :param em_df: DataFrame with PCA generated features
    :param mixture: Ending mixture probabilities for each row and by cluster
    :return df: Loaded data with PCA features and em_classes added on
    """
    classification = []

    # Merge df with em_df
    df = df.merge(em_df[["PCA 1", "PCA 2"]], left_index=True, right_index=True)

    # Assign classes to a list based on highest final mixtures
    for index in range(len(df)):
        if mixture[0][index] > mixture[1][index] and mixture[0][index] > mixture[2][index]:
            classification.append(1)
        elif mixture[1][index] > mixture[0][index] and mixture[1][index] > mixture[2][index]:
            classification.append(2)
        elif mixture[2][index] > mixture[0][index] and mixture[2][index] > mixture[1][index]:
            classification.append(3)

    # Assign classes to df
    df['em_class'] = classification

    return df


def output_expectation_maximization_result(df):
    """
    Method for outputting current DataFrame at the end of this module to a csv

    :param df: Passed in DataFrame for this module
    """
    df.to_csv("output/part6a_expectation_maximization_data.csv")

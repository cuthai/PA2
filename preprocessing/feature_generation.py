import numpy as np
import json


def feature_generation_pca(df, features_to_generate):
    """
    Main method of feature_generation, runs code to generate PCA features off of the four original features

    :param df: loaded data as a DataFrame
    :param features_to_generate: Number of PCA features to return, default is 2
    :return df: Loaded data as a DataFrame with PCA features added on
    """
    # Generate features
    generated_features, eigen_result = generate_features(df, features_to_generate)

    # Add new features onto dataframe
    df = add_to_dataframe(df, generated_features)

    # Output data to a csv
    output_feature_generation_result(df, eigen_result)

    return df


def generate_features(df, features_to_generate, columns=None):
    """
    This method creates test statistics to pass to get_eigen_vectors

    :param df: loaded data as a DataFrame
    :param features_to_generate: Number of PCA features to return, default is 2
    :param columns: Columns to generate PCA features on, default is original 4
    :return (eigen_vectors.T.dot(distances)).T: Filtered eigen vectors times original data
    :return eigen_result: Unfiltered eigen values and vectors as a dictionary
    """
    if columns is None:
        columns = ["sepal length", "sepal width", "petal length", "petal width"]

    # Calculate test statistics
    data_array = df[columns].values
    mean = np.mean(data_array, axis=0)
    covariance = np.cov(data_array.T)

    # Retrieve eigen outcomes from get_eigen_vectors
    eigen_vectors, eigen_result = get_eigen_vectors(covariance, features_to_generate)

    # Calculate distances dot product onto the filtered eigen vectors
    distances = (data_array - mean).T

    return (eigen_vectors.T.dot(distances)).T, eigen_result


def get_eigen_vectors(covariance, features_to_generate):
    """
    Method takes covariance and generates eigen values and vectors using numpy.linalg.eig

    :param covariance: Calculated covariance based on original data
    :param features_to_generate: Number of eigen vectors to return, based on features_to_generate and eigen values
    :return eigen_vectors[:, :features_to_generate]: Filters the top eigen vectors based on eigen values
    :return eigen_result: Unfiltered eigen values and vectors as a dictionary
    """
    # Calculate eigen parameters
    eigen_values, eigen_vectors = np.linalg.eig(covariance)

    # Sort eigen vectors by eigen values
    sort_eigen_values_index = eigen_values.argsort()[::-1]
    eigen_vectors = eigen_vectors[sort_eigen_values_index]

    # Generate a dictionary using the original eigen parameters
    eigen_result = {}
    for index in range(len(eigen_values)):
        eigen_result.update({
            f'Eigen_Value_{eigen_values[index]}': eigen_vectors[index].tolist()
        })

    return eigen_vectors[:, :features_to_generate], eigen_result


def add_to_dataframe(df, generated_features):
    """
    Method to add new features on to original DataFrame

    :param df: loaded data as a DataFrame
    :param generated_features: Generated PCA features
    :return df: Loaded data with the new PCA features
    """
    df["PCA 1"] = generated_features[:, 0]
    df["PCA 2"] = generated_features[:, 1]

    # Reorder the DataFrame so that class is the last column
    df = df[["sepal length", "sepal width", "petal length", "petal width", "New Feature 1", "New Feature 2", "PCA 1",
             "PCA 2", "class"]]

    return df


def output_feature_generation_result(df, eigen_result):
    """
    Method for outputting current DataFrame and eigen parameters at the end of this module to a csv

    :param df: Passed in DataFrame for this module
    :param eigen_result: Unfiltered eigen values and vectors as a dictionary
    """
    df.to_csv("output/part2_feature_generation_data.csv")

    with open("output/part2_feature_generation_algorithm.json", 'w') as file:
        json.dump(eigen_result, file)

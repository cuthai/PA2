from preprocessing.feature_generation import generate_features
import pandas as pd
import json


def dimensionality_reduction_pca(df, features_to_generate):
    """
    Main method for reducing dimensionality down to # of features specified by features_to_generate (default 2)
        Method takes advantage of the generate_features method written in part 3
        The difference here is that this method uses the 4 originals and 2 new for generating PCA features

    :param df: Loaded data as a DataFrame
    :param features_to_generate: Number of features to condense to
    :return generated_df: DataFrame with only the highest scoring PCA features
    """
    # Set columns to the 4 original and 2 new features
    columns = ["sepal length", "sepal width", "petal length", "petal width", "New Feature 1", "New Feature 2"]

    # Generate features
    generated_features, eigen_result = generate_features(df, features_to_generate, columns)

    # Create DataFrame from generated_features with the same indices as the original data
    generated_df = create_dataframe(df, generated_features)

    # Output results to a CSV
    output_dimensionality_reduction_result(generated_df, eigen_result)

    return generated_df


def create_dataframe(df, generated_features):
    """
    Method creates a DataFrame from the generated_features with the same indices as the original data

    :param df: Loaded data as a DataFrame
    :param generated_features: Highest scoring PCA generated features
    :return generated_df: DataFrame with only the highest scoring PCA features
    """
    # Create DataFrame
    generated_df = pd.DataFrame(generated_features)
    generated_df.columns = ['PCA 1', 'PCA 2']

    # Set index equal to original
    generated_df.index = df.index

    return generated_df


def output_dimensionality_reduction_result(df, eigen_result):
    """
    Method for outputting current DataFrame and eigen parameters at the end of this module to a csv

    :param df: Passed in DataFrame for this module
    :param eigen_result: Unfiltered eigen values and vectors as a dictionary
    """
    df.to_csv("output/part5_dimensionality_reduction_data.csv")

    with open("output/part5_dimensionality_reduction_algorithm.json", 'w') as file:
        json.dump(eigen_result, file)

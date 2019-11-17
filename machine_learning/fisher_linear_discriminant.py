import numpy as np


def fisher_linear_discriminant(df):
    """
    Main method for classification using Fisher's LDA
        Uses the four original columns

    :param df: Loaded data as a DataFrame
    """
    # Create copies to avoid modifying the original DataFrames
    df = df.copy(deep=True)

    # Split classes based on original classes
    data_array, data_array_split = split_classes(df)

    # Calculate sw
    sw = calculate_within_class_scatter_matrix(data_array_split)

    # Calculate sb
    sb = calculate_between_class_scatter_matrix(data_array, data_array_split)

    # Use sw and sb to calculate w
    w = calculate_coefficient_w(sw, sb)

    # Create results based on w and the original data
    results = data_array.dot(w)

    # Add classification results to df
    df = classify(df, results)

    # Output data to a CSV
    output_fisher_linear_discriminant_result(df)


def split_classes(df):
    """
    Method splits data based on original classes

    :param df: Loaded data as a DataFrame
    :return data_array: Array with values for all classes
    :return data_array_split: List of arrays with values for each class
    """
    # Create overall array for all classes
    data_array = df[["sepal length", "sepal width", "petal length", "petal width"]].values

    # Create list of arrays for each class
    data_array_split = []
    for index in range(1, 4):
        data_array_split.append(df.loc[df['class'] == index]
                                [["sepal length", "sepal width", "petal length", "petal width"]].values)

    return data_array, data_array_split


def calculate_within_class_scatter_matrix(data_array_split):
    """
    Calculate the within class scatter matrix for each class

    :param data_array_split: List of arrays with values for each class
    :return sw: Total sw matrix summed over all classes
    """
    sw = np.zeros((4, 4))

    # Calculate sw for each class and sum
    for class_data_array in data_array_split:
        separation = (class_data_array.T - class_data_array.mean(axis=0).reshape(4, 1))
        class_sw = separation.dot(separation.T)
        sw += class_sw

    return sw


def calculate_between_class_scatter_matrix(data_array, data_array_split):
    """
    Calculate the between class scatter matrix

    :param data_array: Array with values for all classes
    :param data_array_split: List of arrays with values for each class
    :return sb: Total sb matrix summed over all classes
    """
    sb = np.zeros((4, 4))
    mean = data_array.mean(axis=0)

    # Calculate sb matrix for each class and sum
    for class_data_array in data_array_split:
        class_mean = class_data_array.mean(axis=0)
        distance = (class_mean - mean).reshape(4, 1)
        class_sb = len(class_data_array) * distance.dot(distance.reshape(1, 4))
        sb += class_sb

    return sb


def calculate_coefficient_w(sw, sb):
    """
    Method for calculating w off of sw and sb

    :param sw: Total sw matrix summed over all classes
    :param sb: Total sb matrix summed over all classes
    :return: Two highest scoring eigen vectors form sw and sb
    """
    # Calculate eigen parameters from sw and sb, maximize W
    eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(sw).dot(sb))

    # Sort and return the two highest eigen vectors based off of their eigen values
    sort_eigen_values_index = eigen_values.argsort()[::-1]
    w = eigen_vectors[sort_eigen_values_index][:, :2]

    return w


def classify(df, results):
    """
    Method for classifying data based on highest Fisher's LDA scores

    :param df: Loaded data as a DataFrame
    :param results: Results from Fisher's LDA
    :return: df with flda_class added
    """
    classification = []

    # Create a list of classes based on y
    for row in results:
        if row[0] > 0:
            classification.append(1)
        elif row[0] > -2:
            classification.append(2)
        else:
            classification.append(3)

    # Add classification back to DataFrame
    df['flda_class'] = classification

    return df


def output_fisher_linear_discriminant_result(df):
    """
    Method for outputting current DataFrame at the end of this module to a csv

    :param df: Passed in DataFrame for this module
    """
    df.to_csv("output/part6b_fisher_linear_discriminant_data.csv")

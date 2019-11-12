import numpy as np
import json


def feature_generation_pca(df, features_to_generate):
    generated_features, eigen_result = generate_features(df, features_to_generate)

    df = add_to_dataframe(df, generated_features)

    output_feature_generation_result(df, eigen_result)

    return df


def generate_features(df, features_to_generate, columns=None):
    if columns is None:
        columns = ["sepal length", "sepal width", "petal length", "petal width"]

    data_array = df[columns].values
    mean = np.mean(data_array, axis=0)
    covariance = np.cov(data_array.T)
    eigen_vectors, eigen_result = get_eigen_vectors(covariance, features_to_generate)

    distances = (data_array - mean).T

    return (eigen_vectors.T.dot(distances)).T, eigen_result


def get_eigen_vectors(covariance, features_to_generate):
    eigen_values, eigen_vectors = np.linalg.eig(covariance)

    sort_eigen_values_index = eigen_values.argsort()[::-1]
    eigen_vectors = eigen_vectors[sort_eigen_values_index]

    eigen_result = {}
    for index in range(len(eigen_values)):
        eigen_result.update({
            f'Eigen_Value_{eigen_values[index]}': eigen_vectors[index].tolist()
        })

    return eigen_vectors[:, :features_to_generate], eigen_result


def add_to_dataframe(df, generated_features):
    df["PCA 1"] = generated_features[:, 0]
    df["PCA 2"] = generated_features[:, 1]

    df = df[["sepal length", "sepal width", "petal length", "petal width", "New Feature 1", "New Feature 2", "PCA 1",
             "PCA 2", "class"]]

    return df


def output_feature_generation_result(df, eigen_result):
    df.to_csv("output/part2_feature_generation_data.csv")

    with open("output/part2_feature_generation_algorithm.json", 'w') as file:
        json.dump(eigen_result, file)

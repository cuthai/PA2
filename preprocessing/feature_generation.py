import numpy as np


def feature_generation_pca(df, features_to_generate):
    generated_features = generate_features(df, features_to_generate)

    df = add_to_dataframe(df, generated_features)

    output_feature_generation_result(df)

    return df


def generate_features(df, features_to_generate):
    data_array = df[["sepal length", "sepal width", "petal length", "petal width"]].values
    mean = np.mean(data_array, axis=0)
    covariance = np.cov(data_array.T)
    eigen_vectors = get_eigen_vectors(covariance, features_to_generate)

    distances = (data_array - mean).T

    return (eigen_vectors.T.dot(distances)).T


def get_eigen_vectors(covariance, features_to_generate):
    eigen_values, eigen_vectors = np.linalg.eig(covariance)

    sort_eigen_values_index = eigen_values.argsort()[::-1]
    eigen_vectors = eigen_vectors[sort_eigen_values_index]

    return eigen_vectors[:, :features_to_generate]


def add_to_dataframe(df, generated_features):
    df["PCA 1"] = generated_features[:, 0]
    df["PCA 2"] = generated_features[:, 1]

    df = df[["sepal length", "sepal width", "petal length", "petal width", "New Feature 1", "New Feature 2", "PCA 1",
             "PCA 2", "class"]]

    return df


def output_feature_generation_result(df):
    df.to_csv("output/part2_feature_generation_data.csv", index=False)

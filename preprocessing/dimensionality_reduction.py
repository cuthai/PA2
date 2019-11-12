from preprocessing.feature_generation import generate_features
import pandas as pd
import json


def dimensionality_reduction_pca(df, features_to_generate):
    columns = ["sepal length", "sepal width", "petal length", "petal width", "New Feature 1", "New Feature 2"]
    generated_features, eigen_result = generate_features(df, features_to_generate, columns)

    df = pd.DataFrame(generated_features)
    df.columns = ['PCA 1', 'PCA 2']

    output_dimensionality_reduction_result(df, eigen_result)

    return df


def output_dimensionality_reduction_result(df, eigen_result):
    df.to_csv("output/part5_dimensionality_reduction_data.csv")

    with open("output/part5_dimensionality_reduction_algorithm.json", 'w') as file:
        json.dump(eigen_result, file)

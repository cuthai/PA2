from preprocessing.feature_generation import generate_features
import pandas as pd
import json


def dimensionality_reduction_pca(df, features_to_generate):
    columns = ["sepal length", "sepal width", "petal length", "petal width", "New Feature 1", "New Feature 2"]
    generated_features, eigen_result = generate_features(df, features_to_generate, columns)

    generated_df = pd.DataFrame(generated_features)
    generated_df.columns = ['PCA 1', 'PCA 2']
    generated_df.index = df.index

    output_dimensionality_reduction_result(generated_df, eigen_result)

    return generated_df


def output_dimensionality_reduction_result(df, eigen_result):
    df.to_csv("output/part5_dimensionality_reduction_data.csv")

    with open("output/part5_dimensionality_reduction_algorithm.json", 'w') as file:
        json.dump(eigen_result, file)

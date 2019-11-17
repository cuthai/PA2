import numpy as np


def fisher_linear_discriminant(df):
    df = df.copy(deep=True)

    data_array, data_array_split = split_classes(df)

    sw = calculate_within_class_scatter_matrix(data_array_split)
    sb = calculate_between_class_scatter_matrix(data_array, data_array_split)
    w = calculate_coefficient_w(sw, sb)
    results = data_array.dot(w)

    df = classify(df, results)

    output_fisher_linear_discriminant_result(df)


def split_classes(df):
    data_array = df[["sepal length", "sepal width", "petal length", "petal width"]].values

    data_array_split = []
    for index in range(1, 4):
        data_array_split.append(df.loc[df['class'] == index]
                                [["sepal length", "sepal width", "petal length", "petal width"]].values)

    return data_array, data_array_split


def calculate_within_class_scatter_matrix(data_array_split):
    sw = np.zeros((4, 4))

    for class_data_array in data_array_split:
        separation = (class_data_array.T - class_data_array.mean(axis=0).reshape(4, 1))
        class_sw = separation.dot(separation.T)
        sw += class_sw

    return sw


def calculate_between_class_scatter_matrix(data_array, data_array_split):
    sb = np.zeros((4, 4))
    mean = data_array.mean(axis=0)

    for class_data_array in data_array_split:
        class_mean = class_data_array.mean(axis=0)
        distance = (class_mean - mean).reshape(4, 1)
        class_sb = len(class_data_array) * distance.dot(distance.reshape(1, 4))
        sb += class_sb

    return sb


def calculate_coefficient_w(sw, sb):
    eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(sw).dot(sb))

    sort_eigen_values_index = eigen_values.argsort()[::-1]
    w = eigen_vectors[sort_eigen_values_index][:, :2]

    return w


def classify(df, y):
    classification = []

    for row in y:
        if row[0] > 0:
            classification.append(1)
        elif row[0] > -2:
            classification.append(2)
        else:
            classification.append(3)

    df['flda_class'] = classification

    return df


def output_fisher_linear_discriminant_result(df):
    df.to_csv("output/part6b_fisher_linear_discriminant_data.csv")

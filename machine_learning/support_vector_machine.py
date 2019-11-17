from sklearn import svm


def support_vector_machine(df, pca_df):
    df = df.copy(deep=True)

    results = create_model_and_classify(df, pca_df)

    classify(df, results)

    output_support_vector_machine_result(df)


def create_model_and_classify(df, pca_df):
    x = pca_df[['PCA 1', 'PCA 2']].values
    y = (df[['class']].values - 1).ravel()

    model = svm.LinearSVC(multi_class='ovr')
    model.fit(x, y)

    results = model.predict(x)

    return results.T


def classify(df, results):
    results = results + 1

    df['svm_class'] = results
    return df


def output_support_vector_machine_result(df):
    df.to_csv("output/part6d_support_vector_machine_data.csv")

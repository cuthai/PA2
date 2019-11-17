from sklearn import svm


def support_vector_machine(df, pca_df):
    """
    Main method for classification using a Support Vector Machine
        This method takes advantage of sklearn
        This method does not do any train or test splits because clustering is done over entire data set already

    :param df: Loaded data as a DataFrame
    :param pca_df: DataFrame with PCA generated features
    """
    # Create copies to avoid modifying the original DataFrames
    df = df.copy(deep=True)

    # Create a Support Vector Machine and get results
    results = create_model_and_classify(df, pca_df)

    # Add classification results to df
    classify(df, results)

    # Output data to a CSV
    output_support_vector_machine_result(df)


def create_model_and_classify(df, pca_df):
    """
    This method creates a Support Vector Machine using sklearn

    :param df: Loaded data as a DataFrame
    :param pca_df: DataFrame with PCA generated features
    :return results.T: Results from Support Vector Machine with shape aligned to original data
    """
    # Generate inputs
    x = pca_df[['PCA 1', 'PCA 2']].values
    y = (df[['class']].values - 1).ravel()

    # Create model
    model = svm.LinearSVC(multi_class='ovr')

    # Fit and grab results from model
    model.fit(x, y)
    results = model.predict(x)

    return results.T


def classify(df, results):
    """
    Method for classifying data based on highest Support Vector Machine results

    :param df: Loaded data as a DataFrame
    :param results: Results from Support Vector Machine
    :return: df with svm_class added
    """
    # Since the data was one hot encoded, this adjusts it back similar to the original data
    results = results + 1

    # Add classification back to DataFrame
    df['svm_class'] = results

    return df


def output_support_vector_machine_result(df):
    """
    Method for outputting current DataFrame at the end of this module to a csv

    :param df: Passed in DataFrame for this module
    """
    df.to_csv("output/part6d_support_vector_machine_data.csv")

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import numpy as np


def feed_forward_neural_network(df, pca_df):
    """
    Main method for classification using a Feed Forward Neural Network
        This method takes advantage of Keras
        This method does not do any train or test splits because clustering is done over entire data set already

    :param df: Loaded data as a DataFrame
    :param pca_df: DataFrame with PCA generated features
    """
    # Create copies to avoid modifying the original DataFrames
    df = df.copy(deep=True)

    # Create a Neural Network and get results
    results = create_model_and_classify(df, pca_df)

    # Add classification results to df
    classify(df, results)

    # Output data to a CSV
    output_feed_forward_neural_network_result(df)


def create_model_and_classify(df, pca_df):
    """
    This method creates a single layer neural network in Keras
        The middle layer uses a sigmoid function
        The output layer uses a softmax function

    :param df: Loaded data as a DataFrame
    :param pca_df: DataFrame with PCA generated features
    :return softmax_results: Results from the output layer
    """
    # Generate inputs, y is one hot encoded
    x = pca_df[['PCA 1', 'PCA 2']].values
    y = df[['class']].values - 1
    y = to_categorical(y, num_classes=3)

    # Create model, middle layer uses a sigmoid function and output uses softmax
    model = Sequential([
        Dense(8, input_shape=(2,)),
        Activation('sigmoid'),
        Dense(3),
        Activation('softmax')
    ])

    # Compile based on suggested classification parameters
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Fit and grab softmax results from output
    model.fit(x, y, epochs=1000, batch_size=32)
    softmax_results = model.predict(x)

    return softmax_results


def classify(df, results):
    """
    Method for classifying data based on highest softmax scores

    :param df: Loaded data as a DataFrame
    :param results: Results from Neural Network
    :return: df with ffnn_class added
    """
    classification = []

    # Grab the class with the highest softmax result and append to classification list
    for row in results:
        classification.append(np.where(row == np.amax(row))[0].item())

    # Neural network one hot encoded one of the classes as 0, so this adjusts it back similar to the original data
    classification = np.array(classification) + 1

    # Add classification back to DataFrame
    df['ffnn_class'] = classification

    return df


def output_feed_forward_neural_network_result(df):
    """
    Method for outputting current DataFrame at the end of this module to a csv

    :param df: Passed in DataFrame for this module
    """
    df.to_csv("output/part6c_feed_forward_neural_network_data.csv")

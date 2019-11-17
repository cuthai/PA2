from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import numpy as np


def feed_forward_neural_network(df, pca_df):
    df = df.copy(deep=True)

    results = create_model_and_classify(df, pca_df)

    classify(df, results)

    output_feed_forward_neural_network_result(df)


def create_model_and_classify(df, pca_df):
    x = pca_df[['PCA 1', 'PCA 2']].values
    y = df[['class']].values - 1
    y = to_categorical(y, num_classes=3)

    model = Sequential([
        Dense(8, input_shape=(2,)),
        Activation('sigmoid'),
        Dense(3),
        Activation('softmax')
    ])

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x, y, epochs=1000, batch_size=32)
    softmax_results = model.predict(x)

    return softmax_results


def classify(df, results):
    classification = []

    for row in results:
        classification.append(np.where(row == np.amax(row))[0].item())

    classification = np.array(classification) + 1

    df['ffnn_class'] = classification
    return df


def output_feed_forward_neural_network_result(df):
    df.to_csv("output/part6c_feed_forward_neural_network_data.csv")

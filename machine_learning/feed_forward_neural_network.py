from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical


def feed_forward_neural_network(df):
    x = df[['sepal length', 'sepal width', 'petal length', 'petal width']].values
    y = df[['class']].values - 1
    y = to_categorical(y, num_classes=3)

    model = Sequential([
        Dense(8, input_shape=(4,)),
        Activation('sigmoid'),
        Dense(3),
        Activation('sigmoid')
    ])

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x, y, epochs=1000, batch_size=32)
    results = model.predict(x)

    pass

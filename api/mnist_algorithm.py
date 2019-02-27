import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import Adam

df = pd.read_csv("../resources/mnist.csv")
X = df.drop(["label"], axis=1)
X = X/255
y = df["label"]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.1)


def sequential():
    model = Sequential()

    model.add(Dense(28, input_dim=784, activation=relu))
    model.add(Dense(28, activation=relu))
    model.add(Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X, y, epochs=10, batch_size=10)
    y_pred = model.predict(test_X)
    print(y_pred)

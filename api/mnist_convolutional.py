from tensorflow.python.keras import Sequential
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')

X_train /= 255
X_test /= 255

num_of_classes = 10
y_train = np_utils.to_categorical(y_train, num_of_classes)
y_test = np_utils.to_categorical(y_test, num_of_classes)

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(X_train.shape[1], X_train.shape[2], 1), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_of_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit(X_train, y_train, 200, 10, validation_data=(X_test, y_test))
model.save("../resources/mnist_model2")
metrics = model.evaluate(X_test, y_test, verbose=0)

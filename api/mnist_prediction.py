from tensorflow.python.keras.models import load_model
import numpy as np
from PIL import Image

m = load_model("../resources/mnist_model2")


def predict(input_data):
    in_data = np.array(input_data)
    im = in_data.copy()
    im.resize((28, 28))
    Image.fromarray(im).show()
    in_data = in_data/255
    y_pred = m.predict(in_data)
    return np.argmax(y_pred, axis=1)


def predict2(input_data):
    in_data = np.array(input_data)
    im = in_data.reshape(1, 28, 28, 1)
    y_pred = m.predict(im)
    return np.argmax(y_pred, axis=1)

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

def load_data():
    """Załaduj dane MNIST z Keras."""
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    return (train_images, train_labels), (test_images, test_labels)

def preprocess_data(train_images, train_labels, test_images, test_labels):
    """Znormalizuj obrazy i przekształć etykiety na one-hot encoding."""
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    return train_images, train_labels, test_images, test_labels

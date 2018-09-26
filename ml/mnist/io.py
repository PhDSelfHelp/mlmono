import tensorflow as tf

from ml.base import KerasDatasetIO
from ml.base.trainer_utils import is_training


class MNIST_Keras(KerasDatasetIO):
    _NUM_CLASSES = 10

    @staticmethod
    def keras_load_data(mode):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        if is_training(mode):
            return x_train, y_train
        else:
            return x_test, y_test

    @staticmethod
    def parse_keras_tensor(features, labels):
        return features, labels

    @staticmethod
    def parse_iter(features, label):
        '''Preprocess raw data to trainable input. '''
        x = tf.reshape(tf.cast(features, tf.float32), (28, 28, 1))
        y = tf.one_hot(tf.cast(label, tf.uint8), _NUM_CLASSES)
        return x, y


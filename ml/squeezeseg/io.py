import tensorflow as tf

from ml.base import TFRecordIO
from ml.base.trainer_utils import is_training


class MNIST_Keras(TFRecordIO):

    @staticmethod
    def parse_iter(features, label):
        '''Preprocess raw data to trainable input. '''
        x = tf.reshape(tf.cast(features, tf.float32), (28, 28, 1))
        y = tf.one_hot(tf.cast(label, tf.uint8), _NUM_CLASSES)
        return x, y

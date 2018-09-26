import os
import tensorflow as tf


def find_tfrecords_in_dir(data_dir, file_pattern):
    expr = os.path.join(data_dir, file_pattern)
    return tf.gfile.Glob(expr)


def create_dir_if_not_exist(dirname):
    # os.makedirs does recursive dir creation for dirname if not exist.
    # This does not take a list of dirs as input.
    tf.gfile.MakeDirs(dirname)

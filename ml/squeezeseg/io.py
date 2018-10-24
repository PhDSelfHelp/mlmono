from collections import defaultdict
import itertools
import logging
import operator
import os
import subprocess
import numpy as np
import tensorflow as tf
from tqdm import *

from ml.base import MLConfig, MLIO, TFRecordIO
from ml.base.io_utils import create_dir_if_not_exist
from ml.base.trainer_utils import is_training


_logger = logging.getLogger(name='KittiSqueezeSegIO')


class KittiSqueezeSegIO(TFRecordIO):

    DATASET_URL = "https://www.dropbox.com/s/pnzgcitvppmwfuf/lidar_2d.tgz"
    FULL_NUM_FILES = 10848
    NUM_OF_DATA_GROUPS = 93
    LABEL_TAG = 5
    DEPTH_TAG = 5
    DATA_SHAPE_X = (64, 512, 5)
    DATA_SHAPE_Y = (64, 512, 1)

    INPUT_MEAN = np.array([[[10.88, 0.23, -1.04, 0.21, 12.12]]])
    INPUT_STD = np.array([[[11.47, 6.91, 0.86, 0.16, 12.32]]])

    NUM_CLASS = 4
    CLASSES = ['unknown', 'car', 'pedestrian', 'cyclist']
    CLS_LOSS_WEIGHT = np.array([1/15.0, 1.0, 10.0, 10.0])

    @staticmethod
    def download_data_if_not_exist(data_dir):

        def found_extracted():
            path = os.path.join(data_dir, 'lidar_2d')
            pattern = os.path.join(path, '*.npy')
            return tf.gfile.Exists(path) and \
                   len(tf.gfile.Glob(pattern)) == KittiSqueezeSegIO.FULL_NUM_FILES

        # Download.
        tar_fn = os.path.join(data_dir, 'lidar_2d.tgz')
        if (not tf.gfile.Exists(tar_fn)) and (not found_extracted()):
            result = subprocess.run(['wget', KittiSqueezeSegIO.DATASET_URL, '-P', data_dir],
                                    stdout=subprocess.PIPE)
            _logger.info(result.stdout.decode('utf-8'))

        # Extract.
        if not found_extracted():
            result = subprocess.run(['tar', '-xzvf', tar_fn, '-C', data_dir],
                                    stdout=subprocess.PIPE)
            _logger.info(result.stdout.decode('utf-8'))

        # Remove tar file.
        if tf.gfile.Exists(tar_fn):
            result = subprocess.run(['rm', tar_fn],
                                    stdout=subprocess.PIPE)
            _logger.info(result.stdout.decode('utf-8'))

    @staticmethod
    def parse_file(example_proto):
        features_format = {
            'lidar_input': tf.FixedLenFeature([], tf.float32),
            'lidar_mask' : tf.FixedLenFeature([], tf.float32),
            'weight'     : tf.FixedLenFeature([], tf.float32),
            'labels'     : tf.FixedLenFeature([], tf.float32),
        }
        parsed_features = tf.parse_single_example(example_proto, features_format)
        features = {
            'lidar_input'   : parsed_features['lidar_input'],
            'lidar_mask'    : parsed_features['lidar_mask'],
            'weight'        : parsed_features['weight'],
        }
        labels = tf.sparse_tensor_to_dense(parsed_features['labels'])

        return features, labels

    @staticmethod
    def parse_iter(features, label):
        return features, label

    def create_tf_record(self):
        self.tfrecord_data_dir = os.path.join(self.data_dir, 'lidar_2d_tfrecords')
        create_dir_if_not_exist(self.tfrecord_data_dir)
        _logger.info("Creating tf records : ", self.tfrecord_data_dir)

        numpy_fn_list = tf.gfile.Glob(os.path.join(self.data_dir, '*.npy'))

        for fn in tqdm(numpy_fn_list):
            outpath = os.path.join(self.tfrecord_data_dir, group + ".tfrecords")

            X, Y = self.create_composite_numpy(fn_list)
            np_to_tfrecords(X, Y, outpath)

def _group_numpy_fns(fn_list):
    # example filename: "2011_09_26_0091_0000000062.npy"
    #                          date_ idx_     frame.npy
    groups = defaultdict(list)
    for fn in fn_list:
        group_idx = '_'.join(fn.split('_')[:3])
        group[group_idx].append(fn)
    return groups

def _create_composite_numpy(fn_list):
    X_list = []
    Y_list = []
    for fn in fn_list:
        matrix = np.load(fn)
        X_list.append(matrix[:, :, :KittiSqueezeSegIO.LABEL_TAG])
        Y_list.append(matrix[:, :, KittiSqueezeSegIO.LABEL_TAG])

    composite_X = np.stack(X_list, axis=-1)
    composite_Y = np.stack(Y_list, axis=-1)
    return composite_X, composite_Y

def np_to_tfrecords(X, file_path, verbose=False):
    """ Taken from https://gist.github.com/swyoon/8185b3dcf08ec728fb22b99016dd533f.
        Author: Sangwoong Yoon

    Converts a Numpy array (or two Numpy arrays) into a tfrecord file.
    For supervised learning, feed training inputs to X and training labels to Y.
    For unsupervised learning, only feed training inputs to X, and feed None to Y.
    The length of the first dimensions of X and Y should be the number of samples.

    Parameters
    ----------
    X : numpy.ndarray of rank 2
        Numpy array for training inputs. Its dtype should be float32, float64, or int64.
        If X has a higher rank, it should be rshape before fed to this function.
    Y : numpy.ndarray of rank 2 or None
        Numpy array for training labels. Its dtype should be float32, float64, or int64.
        None if there is no label array.
    file_path : str
        The path and name of the resulting tfrecord file to be generated, with '.tfrecords'
    verbose : bool
        If true, progress is reported.

    Raises
    ------
    ValueError
        If input type is not float (64 or 32) or int.

    """
    def _dtype_feature(ndarray):
        """match appropriate tf.train.Feature class with dtype of ndarray. """
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
        elif dtype_ == np.int64:
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
        else:
            raise ValueError("The input should be numpy ndarray. \
                              Instaed got {}".format(ndarray.dtype))

    assert isinstance(X, np.ndarray)

    # Load appropriate tf.train.Feature class depending on dtype.
    dtype_feature_x = _dtype_feature(X)

    # Generate tfrecord writer.
    writer = tf.python_io.TFRecordWriter(file_path)
    if verbose:
        msg = "Serializing {:d} examples into {}".format(X.shape[0], result_tf_file)
        _logger.info(msg)

    lidar_mask = np.reshape(
        (X[:, :, KittiSqueezeSegIO.DEPTH_TAG] > 0), 
        (self.zenith_level, self.azimuth_level, 1)
    )
    lidar_input = np.reshape(
        X[:, :, :KittiSqueezeSegIO.LABEL_TAG], 
        (self.zenith_level, self.azimuth_level, 1)
    )
    label = X[:, :, :KittiSqueezeSegIO.LABEL_TAG]
    weight = np.zeros(label.shape)
    for l in range(KittiSqueezeSegIO.NUM_CLASS):
        weight[label==l] = KittiSqueezeSegIO.CLS_LOSS_WEIGHT[int(l)]
    d_feature = {
        'lidar_mask': dtype_feature_x(lidar_mask.ravel()),
        'lidar_input': dtype_feature_x(lidar_input.ravel()),
        'label': dtype_feature_x(label.ravel()),
        'weight': dtype_feature_x(weight.ravel()),
    }

    features = tf.train.Features(feature=d_feature)
    example = tf.train.Example(features=features)
    serialized = example.SerializeToString()
    writer.write(serialized)
    writer.close()

    if verbose:
        _logger.info("Writing {} done!".format(result_tf_file))

from collections import defaultdict
import itertools
import logging
import operator
import os
import subprocess
import tensorflow as tf

from ml.base import MLConfig, MLIO, TFRecordIO
from ml.base.io_utils import create_dir_if_not_exist
from ml.base.trainer_utils import is_training
from ml.squeezeseg.io_utils import np_to_tfrecords



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

    CLASSES = ['unknown', 'car', 'pedestrian', 'cyclist']
    CLS_LOSS_WEIGHT = np.array([1/15.0, 1.0, 10.0, 10.0])
    NUM_CLASS = len(KittiSqueezeSegIO.CLASSES)

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
        self.tfrecord_data_dir = os.join(self.data_dir, 'lidar_2d_tfrecords')
        create_dir_if_not_exist(self.tfrecord_data_dir)
        _logger.info("Creating tf records : ", self.tfrecord_data_dir)

        numpy_fn_list = tf.gfile.Glob(os.join(self.data_dir, '*.npy'))

        for fn in tqdm(numpy_fn_list):
            outpath = os.join(self.tfrecord_data_dir, group + ".tfrecords")

            X, Y = self.create_composite_numpy(fn_list)
            np_to_tfrecords(X, Y, outpath)

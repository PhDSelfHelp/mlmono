import logging
import os
import subprocess
import tensorflow as tf

from ml.base import MLConfig, MLIO, TFRecordIO
from ml.base.trainer_utils import is_training


_logger = logging.getLogger(name='KittiSqueezeSegIO')


class KittiSqueezeSegIO(TFRecordIO):

    DATASET_URL = "https://www.dropbox.com/s/pnzgcitvppmwfuf/lidar_2d.tgz"
    FULL_NUM_FILES = 10848

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
    def parse_file(filename):
        return None

    @staticmethod
    def parse_iter(features, label):
        return features, label

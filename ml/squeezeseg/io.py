import logging
import os
import subprocess
import tensorflow as tf

from ml.base import MLConfig, MLIO, TFRecordIO
from ml.base.trainer_utils import is_training


_logger = logging.getLogger(name='KittiSqueezeSegIO')


class KittiSqueezeSegIO(TFRecordIO):

    DATASET_URL = "https://www.dropbox.com/s/pnzgcitvppmwfuf/lidar_2d.tgz"

    @staticmethod
    def download_data_if_not_exist(data_dir):

        def download():
            tar_fn = os.path.join(data_dir, 'lidar_2d.tgz')
            result = subprocess.run(['wget', KittiSqueezeSegIO.DATASET_URL, '-P', data_dir],
                                    stdout=subprocess.PIPE)
            _logger.INFO(result.stdout.decode('utf-8'))
            result = subprocess.run(['tar', '-xzvf', tar_fn, '-C', data_dir],
                                    stdout=subprocess.PIPE)
            _logger.INFO(result.stdout.decode('utf-8'))
            result = subprocess.run(['rm', tar_fn],
                                    stdout=subprocess.PIPE)
            _logger.INFO(result.stdout.decode('utf-8'))

    @staticmethod
    def parse_file(filename):
        return None

    @staticmethod
    def parse_iter(features, label):
        return features, label


config = MLConfig.from_file('ml/squeezeseg/config.yaml')
kitti = MLIO.from_config(config)

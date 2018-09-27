import unittest

from ml.base import MLConfig, MLIO
from ml.squeezeseg.io import KittiSqueezeSegIO


class TestKittiSqueezeSegIO(unittest.TestCase):

    def setUp(self):
        self.config = MLConfig.from_file('ml/squeezeseg/config.yaml')
        self.io = MLIO.from_config(self.config)

    @unittest.skip("Need to download files.")
    def test_download(self):
        self.io.download_data_if_not_exist(kitti.data_dir)

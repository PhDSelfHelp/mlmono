import unittest

from ml.base import MLIO, MLConfig
from ml.mnist.io import MNIST_Keras


class TestMNIST_Keras(unittest.TestCase):
    def setUp(self):
        self.config_dict = {
            'mode': 'train',
            'io': {'io_name': 'MNIST_Keras',
                   'batch_size': 24,
                   'logs_dir': '/tmp/',
                   'model_dir': '/tmp/',
                  },
            'graph': {},
            'metric': {},
            'trainer': {},
            'predictor': {},
        }
        self.mock_config = MLConfig.from_dict(self.config_dict)

    def test_from_config(self):
        mnist_io = MLIO.from_config(self.mock_config)

    def test_gen_input_fn(self):
        num_epochs = 3
        mnist_io = MLIO.from_config(self.mock_config)
        mnist_io.gen_input_fn(3)

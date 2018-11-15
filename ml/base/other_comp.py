from ml.base.graph import MLGraph
from ml.base.metric import MLMetric
from ml.base.io import MLIO
from ml.base.predictor import MLPredictor


class PlaceHolder(MLGraph, MLMetric, MLIO, MLPredictor):
    """ A placeholder class for all of fundamental ML component, it's used for
        matching a component name of 'Placeholder' in config yaml files.
    """

    @classmethod
    def from_config(cls, global_config):
        pass

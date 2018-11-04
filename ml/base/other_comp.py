from ml.base.graph import MLGraph
from ml.base.metric import MLMetric
from ml.base.io import MLIO
from ml.base.predictor import MLPredictor


class PlaceHolder(MLGraph, MLMetric, MLIO, MLPredictor):
    @classmethod
    def from_config(cls, global_config):
        pass

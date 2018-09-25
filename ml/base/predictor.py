import tensorflow as tf


class MLPredictor(object):
    def __init__(self, global_config):
        self.global_config = global_config
        self.predictor_config = global_config.predictor

    @classmethod
    def from_config(cls, global_config):
        predictor_config = global_config.metric
        subcls = find_subclass_by_name(cls, predictor_config.metric_name)
        return subcls.from_config(global_config)

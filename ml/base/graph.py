import tensorflow as tf

from ml.base.base_utils import find_subclass_by_name


class MLGraph(object):

    def __init__(self):
        self.input = None
        self.output = None

    @classmethod
    def from_config(cls, graph_config):
        subcls = find_subclass_by_name(cls, graph_config.model_name)
        return subcls.from_config(graph_config)

    def add_forward_pass(self, input):
        raise NotImplementedError
        return self.output

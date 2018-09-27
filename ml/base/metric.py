from ml.base.base_utils import find_subclass_by_name


class MLMetric(object):

    def __init__(self, global_config):
        self.global_config = global_config
        self.metric_config = self.global_config.metric

    @classmethod
    def from_config(cls, global_config):
        metric_config = global_config.metric
        subcls = find_subclass_by_name(cls, metric_config.metric_name)
        return subcls.from_config(global_config)

    def register_to_graph(self, graph):
        raise NotImplementedError

    def register_to_writer(self, writer):
        raise NotImplementedError

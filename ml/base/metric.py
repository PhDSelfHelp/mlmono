from ml.base.base_utils import find_subclass_by_name


class MLMetric(object):
    def __init__(self):
        pass

    @classmethod
    def from_config(cls, metric_config):
        subcls = find_subclass_by_name(cls, metric_config.metric_name)
        return subcls.from_config(metric_config)

    def register_to_graph(self, graph):
        raise NotImplementedError

    def register_to_writer(self, writer):
        raise NotImplementedError

from ml.base.base_utils import find_subclass_by_name, all_subclasses


class AllMetrics(object):

    def __init__(self, global_config):
        self.global_config = global_config
        self.metric_config = self.global_config.metric

        # Step metrics are evaluated at each step, while 
        self.step_metrics = []
        self.ending_metrics = []

    @classmethod
    def from_config(cls, global_config):
        all_metrics = cls(global_config)
        for metric_config in all_metrics.metric_config:
            subcls = find_subclass_by_name(MLMetric, metric_config.metric_name)

            metric_list = None
            if subcls in all_subclasses(StepMetric):
                metric_list = self.step_metrics
            if subcls in all_subclasses(EndingMetric):
                metric_list = self.ending_metrics

            metric_list.append(subcls.from_config(global_config))
        return all_metrics

    def gen_eval_metric_ops(self):
        ''' Generate every step metrics' `eval_metric_ops` for `tf.estimator.EstimatorSpec`.
        '''
        for metric in self.step_metrics


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


class StepMetric(object):
    pass


class EndingMetric(object):
    pass

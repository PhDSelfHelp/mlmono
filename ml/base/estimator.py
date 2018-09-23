import tensorflow as tf


class MLEstimator(object):
    def __init__(self, config, graph, trainer, evaluator):
        self.config = config
        self.graph = graph
        self.trainer = trainer
        self.evaluator = evaluator
        self.estimator = self._gen_estimator()

    @classmethod
    def from_config(cls, config):
        graph = Graph.from_config(config.graph)
        trainer = Trainer.from_config(config.trainer)
        evaluator = Evaluator.from_config(config.evaluator)
        cls(config, model, trainer, evaluator)

    def _gen_estimator(self):
        estimator = tf.estimator.Estimator(
            model_fn=self._gen_model_fn(),
            input_fn=self._gen_input_fn(),
        )
        return estimator

    def _gen_model_fn(self):
        def model_fn(features, labels, mode, params):
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            result = self.graph.add_forward_pass()


            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op,
                training_hooks=train_hooks,
                eval_metric_ops=metrics)
        return model_fn

    def _gen_input_fn(self):
        def input_fn(features, labels, mode, params):
            return input_fn

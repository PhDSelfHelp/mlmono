import tensorflow as tf


class MLEstimator(object):
    def __init__(self, config, graph, trainer, evaluator, metrics):
        self.config = config
        self.graph = graph
        self.trainer = trainer
        self.evaluator = evaluator
        self.metrics = metrics
        self.estimator = self._gen_estimator()

        self.output = None
        self.labels = None
        self.loss = None

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

            # Construct graph.
            self.output = self.graph.add_forward_pass()

            # Construct trainer.
            self.trainer.register_loss_to_graph(self.output, self.labels)
            self.trainer.register_op_and_hook()
            self.loss = self.trainer.loss

            # Construct metrics.
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=self.ouput,
                loss=self.train,
                train_op=self.trainer.train_op,
                training_hooks=self.trainer.train_hooks,
                eval_metric_ops=metrics)
        return model_fn

    def _gen_input_fn(self):
        def input_fn(features, labels, mode, params):
            return input_fn

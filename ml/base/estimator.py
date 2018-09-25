import tensorflow as tf

from ml.base import MLGraph, MLTrainer


class MLEstimator(object):
    def __init__(self, config, graph, trainer, predictor, metrics, io):
        self.config = config
        self.graph = graph
        self.trainer = trainer
        self.predictor = predictor
        self.metrics = metrics
        self.io = io
        self.estimator = self._gen_estimator()

        self.output = None
        self.labels = None
        self.loss = None

    @classmethod
    def from_config(cls, config):
        graph = MLGraph.from_config(config)
        trainer = MLTrainer.from_config(config)
        predictor = MLPredictor.from_config(config)
        metric = MLMetric.from_config(config)
        io = MLIO.from_config(config)

        return cls(config, model, trainer, predictor, metric, io)

    def _gen_estimator(self):
        estimator = tf.estimator.Estimator(
            model_fn=self._gen_model_fn(),
            input_fn=self._gen_input_fn(),
            model_dir=self.io.model_dir,
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
                eval_metric_ops=metrics
            )
        return model_fn

    def _gen_input_fn(self):
        def input_fn(features, labels, mode, params):
            pass
        return input_fn

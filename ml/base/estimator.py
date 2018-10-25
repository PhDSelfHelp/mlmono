import tensorflow as tf

from ml.base.graph import MLGraph
from ml.base.trainer import MLTrainer
from ml.base.predictor import MLPredictor
from ml.base.metric import MLMetric
from ml.base.io import MLIO


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

        def train():
            self.estimator.train(
                input_fn = self.io.gen_input_fn(self.trainer.num_epochs)
            )

        def predict():
            self.estimator.predict(
                input_fn = self.io.gen_input_fn(self.predictor.num_epochs)
            )

        self.train = train
        self.predict = predict

    @classmethod
    def from_config(cls, config):
        graph = MLGraph.from_config(config)
        trainer = MLTrainer.from_config(config)
        predictor = MLPredictor.from_config(config)
        metric = MLMetric.from_config(config)
        io = MLIO.from_config(config)

        return cls(config, graph, trainer, predictor, metric, io)

    def _gen_estimator(self):
        estimator = tf.estimator.Estimator(
            model_fn=self._gen_model_fn(),
            model_dir=self.io.model_dir,
        )
        return estimator

    def _gen_model_fn(self):

        def model_fn(features, labels, mode, params):
            # Construct graph.
            self.output = self.graph.add_forward_pass()

            # Construct trainer.
            self.trainer.register_loss_to_graph(self.output, self.labels)
            self.trainer.register_op_and_hook()
            self.loss = self.trainer.loss

            # Construct metrics.
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=self.output,
                loss=self.trainer.loss,
                train_op=self.trainer.train_op,
                training_hooks=self.trainer.train_hooks,
                eval_metric_ops=self.metrics)

        return model_fn

    def _gen_input_fn(self):

        def input_fn(features, labels, mode, params):
            pass

        return input_fn

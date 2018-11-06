import tensorflow as tf

from ml.base.graph import MLGraph
from ml.base.trainer import MLTrainer
from ml.base.predictor import MLPredictor
from ml.base.metric import MetricCollection
from ml.base.io import MLIO


class MLEstimator(object):

    def __init__(self, config, graph, trainer, predictor, metric_collection, io):
        self.config = config
        self.graph = graph
        self.trainer = trainer
        self.predictor = predictor
        self.metric_collection = metric_collection
        self.io = io
        self.estimator = self._gen_estimator()

    def train(self):
        print('enter training')
        self.estimator.train(
            input_fn = self.io.gen_input_fn(self.trainer.num_epochs),
            steps=self.config.trainer.num_epochs,
        )

    @classmethod
    def from_config(cls, config):
        graph = MLGraph.from_config(config)
        trainer = MLTrainer.from_config(config)
        predictor = MLPredictor.from_config(config)
        metric_collection = MetricCollection.from_config(config)
        io = MLIO.from_config(config)
        return cls(config, graph, trainer, predictor, metric_collection, io)

    def _gen_estimator(self):
        estimator = tf.estimator.Estimator(
            model_fn=self._gen_model_fn(),
            model_dir=self.io.model_dir,
        )
        return estimator

    def _gen_model_fn(self, gpu_id=2):

        def model_fn(features, labels, mode, params):
            # TODO(jdaaph): Add cpu and gpu flag.
            with tf.device('/gpu:{}'.format(gpu_id)):
                # Construct graph.
                self.graph.add_forward_pass(features, mode)
                self.output = self.graph.output

                # Construct trainer.
                self.trainer.register_loss_to_graph(self.graph, self.output, labels)
                self.trainer.register_train_graph()
                self.trainer.register_op_and_hook()
                self.loss = self.trainer.loss

                # Construct metrics.
                self.metric_collection.register_step_metric_to_graph(self.graph)

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=self.output,
                loss=self.trainer.loss,
                train_op=self.trainer.train_op,
                training_hooks=self.trainer.train_hooks,
                eval_metric_ops={})     # Add metric that needs evaluation each step.
            # TODO(jdaaph): Add tf compatible eval metric ops.

        return model_fn

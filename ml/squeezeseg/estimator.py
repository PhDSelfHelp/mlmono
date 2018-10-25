import tensorflow as tf

from ml.base.estimator import MLEstimator
from ml.squeezeseg.graph import SqueezeSegGraph
from ml.squeezeseg.trainer import SqueezeSegGraph
from ml.squeezeseg.predictor import MLPredictor
from ml.squeezeseg.metric import MLMetric
from ml.squeezeseg.io import KittiSqueezeSegIO


class SqueezeSegEstimator(MLEstimator):
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
            self.trainer.register_train_graph()
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

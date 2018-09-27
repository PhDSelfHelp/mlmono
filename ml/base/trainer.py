import tensorflow as tf

from ml.base import trainer_utils
from ml.base.base_utils import find_subclass_by_name


class MLTrainer(object):

    def __init__(self, global_config):
        self.global_config = global_config
        self.trainer_config = self.global_config.trainer

        self._graph_output = None

        self.train_hooks = []
        self.train_ops = []

        self.loss = None
        self.optimizer = None

    @classmethod
    def from_config(cls, global_config):
        trainer_config = global_config.trainer
        subcls = find_subclass_by_name(cls, trainer_config.trainer_name)
        return subcls.from_config(global_config)

    def register_loss_to_graph(self, graph):
        raise NotImplementedError

    def register_op_and_hook(self):
        raise NotImplementedError


class DefaultTrainer(MLTrainer):

    @classmethod
    def from_config(cls, trainer_config):
        self = cls(trainer_config)

        # Optimizer.
        lr = trainer_config.learning_rate
        if (trainer_config.optimizer).lower() == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        elif (trainer_config.optimizer).lower() == 'sgd':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        return self

    def register_loss_to_graph(self, graph_output, labels):
        # Register loss in this method as it requries to connect to graph.
        self._graph_output = graph_output
        self._labels = labels

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self._graph_output,
                labels=tf.cast(labels, dtype=tf.int32)))

    def register_op_and_hook(self):
        # Train_ops.
        self.train_ops += [
            self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step()),
        ]

        # Train_hooks.
        examples_sec_hook = trainer_utils.ExamplesPerSecondHook(
            params.train_batch_size, every_n_steps=10)
        self.train_hooks += [examples_sec_hook]

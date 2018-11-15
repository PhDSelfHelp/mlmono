import tensorflow as tf

from ml.base.trainer import MLTrainer
from ml.base.base_utils import find_subclass_by_name

class SqueezeSegTrainer(MLTrainer):

    def __init__(self, global_config):
        self.global_config = global_config
        self.trainer_config = self.global_config.trainer

        self._graph_output = None

        self.train_hooks = []
        self.train_ops = []

        self.loss = None
        self.optimizer = None

        # Local config constants.
        self.num_steps = self.trainer_config.num_steps
        self.lr_base = self.trainer_config.lr
        self.decay_steps = self.trainer_config.decay_steps
        self.lr_decay_factor = self.trainer_config.lr_decay_factor
        self.momentum = self.trainer_config.momentum
        self.max_grad_norm = self.trainer_config.max_grad_norm

        self.cls_loss_coef = self.trainer_config.cls_loss_coef

        # Other data related config constants.
        self.num_class = self.global_config.io.num_class

    @property
    def global_step(self):
        return tf.train.get_global_step()

    @classmethod
    def from_config(cls, global_config):
        return cls(global_config)

    def register_op_and_hook(self):
        pass
    
    def register_loss_to_graph(self, graph, graph_output, labels):
        """Define the loss operation."""
        with tf.variable_scope('cls_loss') as scope:
            labels = tf.cast(labels, tf.int32)
            self.cls_loss = tf.identity(
                tf.reduce_sum(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.reshape(labels, (-1, )),
                        logits=tf.reshape(graph_output, (-1, self.num_class))
                    )
                    * tf.reshape(graph.lidar_mask, (-1, ))
                    * tf.reshape(graph.loss_weight, (-1, ))
                ) / tf.reduce_sum(graph.lidar_mask) * self.cls_loss_coef,
                name='cls_loss'
            )
            tf.add_to_collection('losses', self.cls_loss)

        # add above losses as well as weight decay losses to form the total loss
        self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        # add loss summaries
        tf.summary.scalar(self.cls_loss.op.name, self.cls_loss)
        tf.summary.scalar(self.loss.op.name, self.loss)

    def register_train_graph(self):
        """Define the training operation."""

        lr = tf.train.exponential_decay(self.lr_base,
                                        self.global_step,
                                        self.decay_steps,
                                        self.lr_decay_factor,
                                        staircase=True)

        tf.summary.scalar('learning_rate', lr)

        opt = tf.train.MomentumOptimizer(
            learning_rate=lr, momentum=self.momentum)
        grads_vars = opt.compute_gradients(self.loss, tf.trainable_variables())
        with tf.variable_scope('clip_gradient') as scope:
            for i, (grad, var) in enumerate(grads_vars):
                grads_vars[i] = (tf.clip_by_norm(grad, self.max_grad_norm), var)

        apply_gradient_op = opt.apply_gradients(
            grads_vars, global_step=self.global_step)

        self.train_op = apply_gradient_op
        # with tf.control_dependencies([apply_gradient_op]):
        #     self.train_op = tf.no_op(name='train')
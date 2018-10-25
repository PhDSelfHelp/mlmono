
class SqueezeSegTrainer(object):

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

    def register_op_and_hook(self):
        raise NotImplementedError
    
    def register_loss_to_graph(self, graph):
        """Define the loss operation."""
        mc = self.mc

        with tf.variable_scope('cls_loss') as scope:
            self.cls_loss = tf.identity(
                tf.reduce_sum(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.reshape(self.label, (-1, )),
                        logits=tf.reshape(self.output_prob, (-1, mc.NUM_CLASS))
                    )
                    * tf.reshape(self.lidar_mask, (-1, ))
                    * tf.reshape(self.loss_weight, (-1, ))
                )/tf.reduce_sum(self.lidar_mask)*mc.CLS_LOSS_COEF,
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
        mc = self.mc

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        lr = tf.train.exponential_decay(mc.LEARNING_RATE,
                                        self.global_step,
                                        mc.DECAY_STEPS,
                                        mc.LR_DECAY_FACTOR,
                                        staircase=True)

        tf.summary.scalar('learning_rate', lr)

        opt = tf.train.MomentumOptimizer(
            learning_rate=lr, momentum=mc.MOMENTUM)
        grads_vars = opt.compute_gradients(self.loss, tf.trainable_variables())
        with tf.variable_scope('clip_gradient') as scope:
            for i, (grad, var) in enumerate(grads_vars):
                grads_vars[i] = (tf.clip_by_norm(grad, mc.MAX_GRAD_NORM), var)

        apply_gradient_op = opt.apply_gradients(
            grads_vars, global_step=self.global_step)

        with tf.control_dependencies([apply_gradient_op]):
            self.train_op = tf.no_op(name='train')
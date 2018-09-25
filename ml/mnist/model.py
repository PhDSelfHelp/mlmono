import tensorflow.contrib.slim.nets.alexnet as alexnet

from ml.base import MLGraph


class AlexNet(MLGraph):
    def __init__(self, graph_config):
        self.graph_config = graph_config
        self.input = None
        self.output = None

        self.drop_out_keep_prob = self.graph_config.drop_out_keep_prob
        self.global_pool = self.graph_config.global_pool

    @classmethod
    def from_config(cls, graph_config):
        return cls(graph_config)

    def add_forward_pass(self, input, mode):
        self.input = input
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        net, endpoints = alexnet.alexnet_v2(
            self.input,
            num_classes=10,
            is_training=is_training,
            dropout_keep_prob=self.drop_out_keep_prob,
            scope='alexnet_v2',
            global_pool=self.global_pool)
        self.output = [net, endpoints]
        return self.output

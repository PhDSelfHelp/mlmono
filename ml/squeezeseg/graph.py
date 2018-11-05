import os
import sys

import numpy as np
import tensorflow as tf

from ml.base.graph import MLGraph
from ml.squeezeseg.nn_skeleton import ModelSkeleton


class SqueezeSegNet(ModelSkeleton, MLGraph):

    @classmethod
    def from_config(cls, global_config):
        return cls(global_config)

    def __init__(self, global_config):
        self.global_config = global_config
        # TODO(jdaaph): Use an easy to understand method to call ModelSkeleton ctor.
        super().__init__(self.global_config)

        # Initialize constants.
        BATCH_SIZE = self.global_config.trainer.batch_size
        ZENITH_LEVEL = self.global_config.io.zenith_level
        AZIMUTH_LEVEL = self.global_config.io.azimuth_level
        self.classes = self.global_config.io.classes

        # TODO(jdaaph): Clean config hard coded.
        self.LCN_HEIGHT         = 3
        self.LCN_WIDTH          = 5
        self.RCRF_ITER          = 3
        self.BILATERAL_THETA_A  = np.array([.9, .9, .6, .6])
        self.BILATERAL_THETA_R  = np.array([.015, .015, .01, .01])
        self.BI_FILTER_COEF     = 0.1
        self.ANG_THETA_A        = np.array([.9, .9, .6, .6])
        self.ANG_FILTER_COEF = 0.02

        # a scalar tensor in range (0, 1]. Usually set to 0.5 in training phase and
        # 1.0 in evaluation phase
        # self.ph_keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # # projected lidar points on a 2D spherical surface
        # self.ph_lidar_input = tf.placeholder(
        #     tf.float32, [BATCH_SIZE, ZENITH_LEVEL, AZIMUTH_LEVEL, 5],
        #     name='lidar_input'
        # )
        # # A tensor where an element is 1 if the corresponding cell contains an
        # # valid lidar measurement. Or if the data is missing, then mark it as 0.
        # self.ph_lidar_mask = tf.placeholder(
        #     tf.float32, [BATCH_SIZE, ZENITH_LEVEL, AZIMUTH_LEVEL, 1],
        #     name='lidar_mask')
        # # A tensor where each element contains the class of each pixel
        # self.ph_label = tf.placeholder(
        #     tf.int32, [BATCH_SIZE, ZENITH_LEVEL, AZIMUTH_LEVEL],
        #     name='label')
        # # weighted loss for different classes
        # self.ph_loss_weight = tf.placeholder(
        #     tf.float32, [BATCH_SIZE, ZENITH_LEVEL, AZIMUTH_LEVEL],
        #     name='loss_weight')

        # # define a FIFOqueue for pre-fetching data
        # self.q = tf.FIFOQueue(
        #     capacity=mc.QUEUE_CAPACITY,
        #     dtypes=[tf.float32, tf.float32, tf.float32, tf.int32, tf.float32],
        #     shapes=[[],
        #             [BATCH_SIZE, ZENITH_LEVEL, AZIMUTH_LEVEL, 5],
        #             [BATCH_SIZE, ZENITH_LEVEL, AZIMUTH_LEVEL, 1],
        #             [BATCH_SIZE, ZENITH_LEVEL, AZIMUTH_LEVEL],
        #             [BATCH_SIZE, ZENITH_LEVEL, AZIMUTH_LEVEL]]
        # )
        # self.enqueue_op = self.q.enqueue(
        #     [self.ph_keep_prob, self.ph_lidar_input, self.ph_lidar_mask,
        #      self.ph_label, self.ph_loss_weight]
        # )

        # model parameters
        self.model_params = []

        # model size counter
        self.model_size_counter = []  # array of tuple of layer name, parameter size
        # flop counter
        self.flop_counter = []  # array of tuple of layer name, flop number
        # activation counter
        self.activation_counter = []  # array of tuple of layer name, output activations
        self.activation_counter.append(
            ('input', AZIMUTH_LEVEL * ZENITH_LEVEL * 3))

    def _add_output_graph(self):
        """Define how to intepret output."""

        with tf.variable_scope('interpret_output') as scope:
            self.prob = tf.multiply(
                tf.nn.softmax(self.output_prob, dim=-1), self.lidar_mask,
                name='pred_prob')
            self.pred_cls = tf.argmax(self.prob, axis=3, name='pred_cls')

            # Add activation summaries.
            for cls_id, cls_name in enumerate(self.classes):
                self._activation_summary(
                    self.prob[:, :, :, cls_id], 'prob_' + cls_name)

            self.output = self.prob

    def add_forward_pass(self, features, mode):
        """NN architecture."""

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # Input parsing for the graph.
        self.keep_prob = 0.5 if is_training else 1
        self.lidar_input = features['lidar_input']
        self.lidar_mask = features['lidar_mask']
        self.loss_weight = features['weight']
        
        conv1 = self._conv_layer(
            'conv1', self.lidar_input, filters=64, size=3, stride=2,
            padding='SAME', freeze=False, xavier=True)
        conv1_skip = self._conv_layer(
            'conv1_skip', self.lidar_input, filters=64, size=1, stride=1,
            padding='SAME', freeze=False, xavier=True)
        pool1 = self._pooling_layer(
            'pool1', conv1, size=3, stride=2, padding='SAME')

        fire2 = self._fire_layer(
            'fire2', pool1, s1x1=16, e1x1=64, e3x3=64, freeze=False)
        fire3 = self._fire_layer(
            'fire3', fire2, s1x1=16, e1x1=64, e3x3=64, freeze=False)
        pool3 = self._pooling_layer(
            'pool3', fire3, size=3, stride=2, padding='SAME')

        fire4 = self._fire_layer(
            'fire4', pool3, s1x1=32, e1x1=128, e3x3=128, freeze=False)
        fire5 = self._fire_layer(
            'fire5', fire4, s1x1=32, e1x1=128, e3x3=128, freeze=False)
        pool5 = self._pooling_layer(
            'pool5', fire5, size=3, stride=2, padding='SAME')

        fire6 = self._fire_layer(
            'fire6', pool5, s1x1=48, e1x1=192, e3x3=192, freeze=False)
        fire7 = self._fire_layer(
            'fire7', fire6, s1x1=48, e1x1=192, e3x3=192, freeze=False)
        fire8 = self._fire_layer(
            'fire8', fire7, s1x1=64, e1x1=256, e3x3=256, freeze=False)
        fire9 = self._fire_layer(
            'fire9', fire8, s1x1=64, e1x1=256, e3x3=256, freeze=False)

        # Deconvolation
        fire10 = self._fire_deconv(
            'fire_deconv10', fire9, s1x1=64, e1x1=128, e3x3=128, factors=[1, 2],
            stddev=0.1)
        fire10_fuse = tf.add(fire10, fire5, name='fure10_fuse')

        fire11 = self._fire_deconv(
            'fire_deconv11', fire10_fuse, s1x1=32, e1x1=64, e3x3=64, factors=[1, 2],
            stddev=0.1)
        fire11_fuse = tf.add(fire11, fire3, name='fire11_fuse')

        fire12 = self._fire_deconv(
            'fire_deconv12', fire11_fuse, s1x1=16, e1x1=32, e3x3=32, factors=[1, 2],
            stddev=0.1)
        fire12_fuse = tf.add(fire12, conv1, name='fire12_fuse')

        fire13 = self._fire_deconv(
            'fire_deconv13', fire12_fuse, s1x1=16, e1x1=32, e3x3=32, factors=[1, 2],
            stddev=0.1)
        fire13_fuse = tf.add(fire13, conv1_skip, name='fire13_fuse')

        drop13 = tf.nn.dropout(fire13_fuse, self.keep_prob, name='drop13')

        conv14 = self._conv_layer(
            'conv14_prob', drop13, filters=self.NUM_CLASS, size=3, stride=1,
            padding='SAME', relu=False, stddev=0.1)

        bilateral_filter_weights = self._bilateral_filter_layer(
            'bilateral_filter', self.lidar_input[:, :, :, :3],  # x, y, z
            thetas=[self.BILATERAL_THETA_A, self.BILATERAL_THETA_R],
            sizes=[self.LCN_HEIGHT, self.LCN_WIDTH], stride=1)

        self.output_prob = self._recurrent_crf_layer(
            'recurrent_crf', conv14, bilateral_filter_weights,
            sizes=[self.LCN_HEIGHT, self.LCN_WIDTH], num_iterations=self.RCRF_ITER,
            padding='SAME'
        )
        self._add_output_graph()

        return self.output_prob

    def _fire_layer(self, layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.001,
                    freeze=False):
        """Fire layer constructor.
        Args:
        layer_name: layer name
        inputs: input tensor
        s1x1: number of 1x1 filters in squeeze layer.
        e1x1: number of 1x1 filters in expand layer.
        e3x3: number of 3x3 filters in expand layer.
        freeze: if true, do not train parameters in this layer.
        Returns:
        fire layer operation.
        """

        sq1x1 = self._conv_layer(
            layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
            padding='SAME', freeze=freeze, stddev=stddev)
        ex1x1 = self._conv_layer(
            layer_name+'/expand1x1', sq1x1, filters=e1x1, size=1, stride=1,
            padding='SAME', freeze=freeze, stddev=stddev)
        ex3x3 = self._conv_layer(
            layer_name+'/expand3x3', sq1x1, filters=e3x3, size=3, stride=1,
            padding='SAME', freeze=freeze, stddev=stddev)

        return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')

    def _fire_deconv(self, layer_name, inputs, s1x1, e1x1, e3x3,
                     factors=[1, 2], freeze=False, stddev=0.001):
        """Fire deconvolution layer constructor.
        Args:
        layer_name: layer name
        inputs: input tensor
        s1x1: number of 1x1 filters in squeeze layer.
        e1x1: number of 1x1 filters in expand layer.
        e3x3: number of 3x3 filters in expand layer.
        factors: spatial upsampling factors.
        freeze: if true, do not train parameters in this layer.
        Returns:
        fire layer operation.
        """

        assert len(factors) == 2, 'factors should be an array of size 2'

        ksize_h = factors[0] * 2 - factors[0] % 2
        ksize_w = factors[1] * 2 - factors[1] % 2

        sq1x1 = self._conv_layer(
            layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
            padding='SAME', freeze=freeze, stddev=stddev)
        deconv = self._deconv_layer(
            layer_name+'/deconv', sq1x1, filters=s1x1, size=[ksize_h, ksize_w],
            stride=factors, padding='SAME', init='bilinear')
        ex1x1 = self._conv_layer(
            layer_name+'/expand1x1', deconv, filters=e1x1, size=1, stride=1,
            padding='SAME', freeze=freeze, stddev=stddev)
        ex3x3 = self._conv_layer(
            layer_name+'/expand3x3', deconv, filters=e3x3, size=3, stride=1,
            padding='SAME', freeze=freeze, stddev=stddev)

        return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')

    def _activation_summary(self, x, layer_name):
        """Helper to create summaries for activations.

        Args:
          x: layer output tensor
          layer_name: name of the layer
        Returns:
          nothing
        """
        with tf.variable_scope('activation_summary') as scope:
            tf.summary.histogram(layer_name, x)
            tf.summary.scalar(layer_name+'/sparsity', tf.nn.zero_fraction(x))
            tf.summary.scalar(layer_name+'/average', tf.reduce_mean(x))
            tf.summary.scalar(layer_name+'/max', tf.reduce_max(x))
            tf.summary.scalar(layer_name+'/min', tf.reduce_min(x))

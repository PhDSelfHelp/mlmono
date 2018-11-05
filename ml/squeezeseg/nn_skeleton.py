import os
import sys

import numpy as np
import tensorflow as tf

from ml.squeezeseg import utils


def _variable_on_device(name, shape, initializer, trainable=True):
    """Helper to create a Variable.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    # TODO(bichen): fix the hard-coded data type below
    dtype = tf.float32
    if not callable(initializer):
        var = tf.get_variable(
            name, initializer=initializer, trainable=trainable)
    else:
        var = tf.get_variable(
            name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    return var


def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    var = _variable_on_device(name, shape, initializer, trainable)
    if wd is not None and trainable:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


class ModelSkeleton(object):
    """Base class of NN detection components."""

    def __init__(self, global_config):
        self.LOAD_PRETRAINED_MODEL = False
        self.WEIGHT_DECAY = global_config.trainer.weight_decay
        self.BATCH_NORM_EPSILON = 1e-5
        self.DEBUG_MODE = True
        self.BATCH_SIZE = global_config.trainer.batch_size
        self.NUM_CLASS = global_config.io.num_class
        self.BI_FILTER_COEF = global_config.graph.bi_filter_coef

    def _conv_bn_layer(
            self, inputs, conv_param_name, bn_param_name, scale_param_name, filters,
            size, stride, padding='SAME', freeze=False, relu=True,
            conv_with_bias=False, stddev=0.001):
        """ Convolution + BatchNorm + [relu] layer. Batch mean and var are treated
        as constant. Weights have to be initialized from a pre-trained model or
        restored from a checkpoint.

        Args:
          inputs: input tensor
          conv_param_name: name of the convolution parameters
          bn_param_name: name of the batch normalization parameters
          scale_param_name: name of the scale parameters
          filters: number of output filters.
          size: kernel size.
          stride: stride
          padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
          freeze: if true, then do not train the parameters in this layer.
          xavier: whether to use xavier weight initializer or not.
          relu: whether to use relu or not.
          conv_with_bias: whether or not add bias term to the convolution output.
          stddev: standard deviation used for random weight initializer.
        Returns:
          A convolutional layer operation.
        """

        with tf.variable_scope(conv_param_name) as scope:
            channels = inputs.get_shape()[3]

            if self.LOAD_PRETRAINED_MODEL:
                cw = self.caffemodel_weight
                kernel_val = np.transpose(cw[conv_param_name][0], [2, 3, 1, 0])
                if conv_with_bias:
                    bias_val = cw[conv_param_name][1]
                mean_val = cw[bn_param_name][0]
                var_val = cw[bn_param_name][1]
                gamma_val = cw[scale_param_name][0]
                beta_val = cw[scale_param_name][1]
            else:
                kernel_val = tf.truncated_normal_initializer(
                    stddev=stddev, dtype=tf.float32)
                if conv_with_bias:
                    bias_val = tf.constant_initializer(0.0)
                mean_val = tf.constant_initializer(0.0)
                var_val = tf.constant_initializer(1.0)
                gamma_val = tf.constant_initializer(1.0)
                beta_val = tf.constant_initializer(0.0)

            # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
            # shape [h, w, in, out]
            kernel = _variable_with_weight_decay(
                'kernels', shape=[size, size, int(channels), filters],
                wd=self.WEIGHT_DECAY, initializer=kernel_val, trainable=(not freeze))
            self.model_params += [kernel]
            if conv_with_bias:
                biases = _variable_on_device('biases', [filters], bias_val,
                                             trainable=(not freeze))
                self.model_params += [biases]
            gamma = _variable_on_device('gamma', [filters], gamma_val,
                                        trainable=(not freeze))
            beta = _variable_on_device('beta', [filters], beta_val,
                                       trainable=(not freeze))
            mean = _variable_on_device(
                'mean', [filters], mean_val, trainable=False)
            var = _variable_on_device(
                'var', [filters], var_val, trainable=False)
            self.model_params += [gamma, beta, mean, var]

            conv = tf.nn.conv2d(
                inputs, kernel, [1, 1, stride, 1], padding=padding,
                name='convolution')
            if conv_with_bias:
                conv = tf.nn.bias_add(conv, biases, name='bias_add')

            conv = tf.nn.batch_normalization(
                conv, mean=mean, variance=var, offset=beta, scale=gamma,
                variance_epsilon=self.BATCH_NORM_EPSILON, name='batch_norm')

            self.model_size_counter.append(
                (conv_param_name, (1 + size * size * int(channels)) * filters)
            )
            out_shape = conv.get_shape().as_list()
            num_flops = \
                (1 + 2 * int(channels) * size * size) * filters * out_shape[1] * out_shape[2]
            if relu:
                num_flops += 2 * filters * out_shape[1] * out_shape[2]
            self.flop_counter.append((conv_param_name, num_flops))

            self.activation_counter.append(
                (conv_param_name, out_shape[1] * out_shape[2] * out_shape[3])
            )

            if relu:
                return tf.nn.relu(conv)
            else:
                return conv

    def _conv_layer(
            self, layer_name, inputs, filters, size, stride, padding='SAME',
            freeze=False, xavier=False, relu=True, stddev=0.001, bias_init_val=0.0):
        """Convolutional layer operation constructor.

        Args:
          layer_name: layer name.
          inputs: input tensor
          filters: number of output filters.
          size: kernel size.
          stride: stride
          padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
          freeze: if true, then do not train the parameters in this layer.
          xavier: whether to use xavier weight initializer or not.
          relu: whether to use relu or not.
          stddev: standard deviation used for random weight initializer.
        Returns:
          A convolutional layer operation.
        """

        use_pretrained_param = False
        if self.LOAD_PRETRAINED_MODEL:
            cw = self.caffemodel_weight
            if layer_name in cw:
                kernel_val = np.transpose(cw[layer_name][0], [2, 3, 1, 0])
                bias_val = cw[layer_name][1]
                # check the shape
                if (kernel_val.shape ==
                    (size, size, inputs.get_shape().as_list()[-1], filters)) \
                   and (bias_val.shape == (filters, )):
                    use_pretrained_param = True
                else:
                    print('Shape of the pretrained parameter of {} does not match, '
                          'use randomly initialized parameter'.format(layer_name))
            else:
                print('Cannot find {} in the pretrained model. Use randomly initialized '
                      'parameters'.format(layer_name))

        if self.DEBUG_MODE:
            print('Input tensor shape to {}: {}'.format(
                layer_name, inputs.get_shape()))

        with tf.variable_scope(layer_name) as scope:
            channels = inputs.get_shape()[3]

            # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
            # shape [h, w, in, out]
            if use_pretrained_param:
                if self.DEBUG_MODE:
                    print('Using pretrained model for {}'.format(layer_name))
                kernel_init = tf.constant(kernel_val, dtype=tf.float32)
                bias_init = tf.constant(bias_val, dtype=tf.float32)
            elif xavier:
                kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
                bias_init = tf.constant_initializer(bias_init_val)
            else:
                kernel_init = tf.truncated_normal_initializer(
                    stddev=stddev, dtype=tf.float32)
                bias_init = tf.constant_initializer(bias_init_val)

            kernel = _variable_with_weight_decay(
                'kernels', shape=[size, size, int(channels), filters],
                wd=self.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))

            biases = _variable_on_device('biases', [filters], bias_init,
                                         trainable=(not freeze))
            self.model_params += [kernel, biases]

            conv = tf.nn.conv2d(
                inputs, kernel, [1, 1, stride, 1], padding=padding,
                name='convolution')
            conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')

            if relu:
                out = tf.nn.relu(conv_bias, 'relu')
            else:
                out = conv_bias

            self.model_size_counter.append(
                (layer_name, (1 + size * size * int(channels)) * filters)
            )
            out_shape = out.get_shape().as_list()
            num_flops = \
                (1 + 2 * int(channels) * size * size) * filters * out_shape[1] * out_shape[2]
            if relu:
                num_flops += 2 * filters * out_shape[1] * out_shape[2]
            self.flop_counter.append((layer_name, num_flops))

            self.activation_counter.append(
                (layer_name, out_shape[1] * out_shape[2] * out_shape[3])
            )

            return out

    def _deconv_layer(
            self, layer_name, inputs, filters, size, stride, padding='SAME',
            freeze=False, init='trunc_norm', relu=True, stddev=0.001):
        """Deconvolutional layer operation constructor.

        Args:
          layer_name: layer name.
          inputs: input tensor
          filters: number of output filters.
          size: kernel size. An array of size 2 or 1.
          stride: stride. An array of size 2 or 1.
          padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
          freeze: if true, then do not train the parameters in this layer.
          init: how to initialize kernel weights. Now accept 'xavier',
              'trunc_norm', 'bilinear'
          relu: whether to use relu or not.
          stddev: standard deviation used for random weight initializer.
        Returns:
          A convolutional layer operation.
        """

        assert len(size) == 1 or len(size) == 2, \
            'size should be a scalar or an array of size 2.'
        assert len(stride) == 1 or len(stride) == 2, \
            'stride should be a scalar or an array of size 2.'
        assert init == 'xavier' or init == 'bilinear' or init == 'trunc_norm', \
            'initi mode not supported {}'.format(init)

        if len(size) == 1:
            size_h, size_w = size[0], size[0]
        else:
            size_h, size_w = size[0], size[1]

        if len(stride) == 1:
            stride_h, stride_w = stride[0], stride[0]
        else:
            stride_h, stride_w = stride[0], stride[1]

        # TODO(bichen): Currently do not support pretrained parameters for deconv
        # layer.

        if self.DEBUG_MODE:
            print('Input tensor shape to {}: {}'.format(
                layer_name, inputs.get_shape()))

        with tf.variable_scope(layer_name) as scope:
            in_height = int(inputs.get_shape()[1])
            in_width = int(inputs.get_shape()[2])
            channels = int(inputs.get_shape()[3])

            if init == 'xavier':
                kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
                bias_init = tf.constant_initializer(0.0)
            elif init == 'bilinear':
                assert size_h == 1, 'Now only support size_h=1'
                assert channels == filters, \
                    'In bilinear interporlation mode, input channel size and output' \
                    'filter size should be the same'
                assert stride_h == 1, \
                    'In bilinear interpolation mode, stride_h should be 1'

                kernel_init = np.zeros(
                    (size_h, size_w, channels, channels),
                    dtype=np.float32)

                factor_w = (size_w + 1)//2
                assert factor_w == stride_w, \
                    'In bilinear interpolation mode, stride_w == factor_w'

                center_w = (factor_w - 1) if (size_w %
                                              2 == 1) else (factor_w - 0.5)
                og_w = np.reshape(np.arange(size_w), (size_h, -1))
                up_kernel = (1 - np.abs(og_w - center_w)/factor_w)
                for c in range(channels):
                    kernel_init[:, :, c, c] = up_kernel

                bias_init = tf.constant_initializer(0.0)
            else:
                kernel_init = tf.truncated_normal_initializer(
                    stddev=stddev, dtype=tf.float32)
                bias_init = tf.constant_initializer(0.0)

            # Kernel layout for deconv layer: [H_f, W_f, O_c, I_c] where I_c is the
            # input channel size. It should be the same as the channel size of the
            # input tensor.
            kernel = _variable_with_weight_decay(
                'kernels', shape=[size_h, size_w, filters, channels],
                wd=self.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))
            biases = _variable_on_device(
                'biases', [filters], bias_init, trainable=(not freeze))
            self.model_params += [kernel, biases]

            # TODO(bichen): fix this
            deconv = tf.nn.conv2d_transpose(
                inputs, kernel,
                [self.BATCH_SIZE, stride_h*in_height, stride_w*in_width, filters],
                [1, stride_h, stride_w, 1], padding=padding,
                name='deconv')
            deconv_bias = tf.nn.bias_add(deconv, biases, name='bias_add')

            if relu:
                out = tf.nn.relu(deconv_bias, 'relu')
            else:
                out = deconv_bias

            self.model_size_counter.append(
                (layer_name, (1+size_h*size_w*channels)*filters)
            )
            out_shape = out.get_shape().as_list()
            num_flops = \
                (1 + 2 * channels * size_h * size_w) * filters \
                * out_shape[1] * out_shape[2]
            if relu:
                num_flops += 2 * filters * out_shape[1] * out_shape[2]
            self.flop_counter.append((layer_name, num_flops))

            self.activation_counter.append(
                (layer_name, out_shape[1] * out_shape[2] * out_shape[3])
            )

            return out

    def _pooling_layer(
            self, layer_name, inputs, size, stride, padding='SAME'):
        """Pooling layer operation constructor.

        Args:
          layer_name: layer name.
          inputs: input tensor
          size: kernel size.
          stride: stride
          padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
        Returns:
          A pooling layer operation.
        """

        with tf.variable_scope(layer_name) as scope:
            out = tf.nn.max_pool(inputs,
                                 ksize=[1, size, size, 1],
                                 strides=[1, 1, stride, 1],
                                 padding=padding)
            activation_size = np.prod(out.get_shape().as_list()[1:])
            self.activation_counter.append((layer_name, activation_size))
            return out

    def _fc_layer(
            self, layer_name, inputs, hiddens, flatten=False, relu=True,
            xavier=False, stddev=0.001, bias_init_val=0.0):
        """Fully connected layer operation constructor.

        Args:
          layer_name: layer name.
          inputs: input tensor
          hiddens: number of (hidden) neurons in this layer.
          flatten: if true, reshape the input 4D tensor of shape 
              (batch, height, weight, channel) into a 2D tensor with shape 
              (batch, -1). This is used when the input to the fully connected layer
              is output of a convolutional layer.
          relu: whether to use relu or not.
          xavier: whether to use xavier weight initializer or not.
          stddev: standard deviation used for random weight initializer.
        Returns:
          A fully connected layer operation.
        """

        use_pretrained_param = False
        if self.LOAD_PRETRAINED_MODEL:
            cw = self.caffemodel_weight
            if layer_name in cw:
                use_pretrained_param = True
                kernel_val = cw[layer_name][0]
                bias_val = cw[layer_name][1]

        if self.DEBUG_MODE:
            print('Input tensor shape to {}: {}'.format(
                layer_name, inputs.get_shape()))

        with tf.variable_scope(layer_name) as scope:
            input_shape = inputs.get_shape().as_list()
            if flatten:
                dim = input_shape[1]*input_shape[2]*input_shape[3]
                inputs = tf.reshape(inputs, [-1, dim])
                if use_pretrained_param:
                    try:
                        # check the size before layout transform
                        assert kernel_val.shape == (hiddens, dim), \
                            'kernel shape error at {}'.format(layer_name)
                        kernel_val = np.reshape(
                            np.transpose(
                                np.reshape(
                                    kernel_val,  # O x (C*H*W)
                                    (hiddens, input_shape[3],
                                     input_shape[1], input_shape[2])
                                ),  # O x C x H x W
                                (2, 3, 1, 0)
                            ),  # H x W x C x O
                            (dim, -1)
                        )  # (H*W*C) x O
                        # check the size after layout transform
                        assert kernel_val.shape == (dim, hiddens), \
                            'kernel shape error at {}'.format(layer_name)
                    except:
                        # Do not use pretrained parameter if shape doesn't match
                        use_pretrained_param = False
                        print('Shape of the pretrained parameter of {} does not match, '
                              'use randomly initialized parameter'.format(layer_name))
            else:
                dim = input_shape[1]
                if use_pretrained_param:
                    try:
                        kernel_val = np.transpose(kernel_val, (1, 0))
                        assert kernel_val.shape == (dim, hiddens), \
                            'kernel shape error at {}'.format(layer_name)
                    except:
                        use_pretrained_param = False
                        print('Shape of the pretrained parameter of {} does not match, '
                              'use randomly initialized parameter'.format(layer_name))

            if use_pretrained_param:
                if self.DEBUG_MODE:
                    print('Using pretrained model for {}'.format(layer_name))
                kernel_init = tf.constant(kernel_val, dtype=tf.float32)
                bias_init = tf.constant(bias_val, dtype=tf.float32)
            elif xavier:
                kernel_init = tf.contrib.layers.xavier_initializer()
                bias_init = tf.constant_initializer(bias_init_val)
            else:
                kernel_init = tf.truncated_normal_initializer(
                    stddev=stddev, dtype=tf.float32)
                bias_init = tf.constant_initializer(bias_init_val)

            weights = _variable_with_weight_decay(
                'weights', shape=[dim, hiddens], wd=self.WEIGHT_DECAY,
                initializer=kernel_init)
            biases = _variable_on_device('biases', [hiddens], bias_init)
            self.model_params += [weights, biases]

            outputs = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
            if relu:
                outputs = tf.nn.relu(outputs, 'relu')

            # count layer stats
            self.model_size_counter.append((layer_name, (dim+1)*hiddens))

            num_flops = 2 * dim * hiddens + hiddens
            if relu:
                num_flops += 2*hiddens
            self.flop_counter.append((layer_name, num_flops))

            self.activation_counter.append((layer_name, hiddens))

            return outputs

    def _recurrent_crf_layer(
            self, layer_name, inputs, bilateral_filters, sizes=[3, 5],
            num_iterations=1, padding='SAME'):
        """Recurrent conditional random field layer. Iterative meanfield inference is
        implemented as a reccurent neural network.

        Args:
          layer_name: layer name
          inputs: input tensor with shape [batch_size, zenith, azimuth, num_class].
          bilateral_filters: filter weight with shape 
              [batch_size, zenith, azimuth, sizes[0]*size[1]-1].
          sizes: size of the local region to be filtered.
          num_iterations: number of meanfield inferences.
          padding: padding strategy
        Returns:
          outputs: tensor with shape [batch_size, zenith, azimuth, num_class].
        """
        assert num_iterations >= 1, 'number of iterations should >= 1'

        with tf.variable_scope(layer_name) as scope:
            # initialize compatibilty matrices
            compat_kernel_init = tf.constant(
                np.reshape(
                    np.ones((self.NUM_CLASS, self.NUM_CLASS)) -
                    np.identity(self.NUM_CLASS),
                    [1, 1, self.NUM_CLASS, self.NUM_CLASS]
                ),
                dtype=tf.float32
            )
            bi_compat_kernel = _variable_on_device(
                name='bilateral_compatibility_matrix',
                shape=[1, 1, self.NUM_CLASS, self.NUM_CLASS],
                initializer=compat_kernel_init*self.BI_FILTER_COEF,
                trainable=True
            )
            self._activation_summary(bi_compat_kernel, 'bilateral_compat_mat')

            angular_compat_kernel = _variable_on_device(
                name='angular_compatibility_matrix',
                shape=[1, 1, self.NUM_CLASS, self.NUM_CLASS],
                initializer=compat_kernel_init * self.ANG_FILTER_COEF,
                trainable=True
            )
            self._activation_summary(
                angular_compat_kernel, 'angular_compat_mat')

            self.model_params += [bi_compat_kernel, angular_compat_kernel]

            condensing_kernel = tf.constant(
                utils.condensing_matrix(sizes[0], sizes[1], self.NUM_CLASS),
                dtype=tf.float32,
                name='condensing_kernel'
            )

            angular_filters = tf.constant(
                utils.angular_filter_kernel(
                    sizes[0], sizes[1], self.NUM_CLASS, self.ANG_THETA_A**2),
                dtype=tf.float32,
                name='angular_kernel'
            )

            bi_angular_filters = tf.constant(
                utils.angular_filter_kernel(
                    sizes[0], sizes[1], self.NUM_CLASS, self.BILATERAL_THETA_A**2),
                dtype=tf.float32,
                name='bi_angular_kernel'
            )

            for it in range(num_iterations):
                unary = tf.nn.softmax(
                    inputs, dim=-1, name='unary_term_at_iter_{}'.format(it))

                ang_output, bi_output = self._locally_connected_layer(
                    'message_passing_iter_{}'.format(it), unary,
                    bilateral_filters, angular_filters, bi_angular_filters,
                    condensing_kernel, sizes=sizes,
                    padding=padding
                )

                # 1x1 convolution as compatibility transform
                ang_output = tf.nn.conv2d(
                    ang_output, angular_compat_kernel, strides=[1, 1, 1, 1],
                    padding='SAME', name='angular_compatibility_transformation')
                self._activation_summary(
                    ang_output, 'ang_transfer_iter_{}'.format(it))

                bi_output = tf.nn.conv2d(
                    bi_output, bi_compat_kernel, strides=[1, 1, 1, 1], padding='SAME',
                    name='bilateral_compatibility_transformation')
                self._activation_summary(
                    bi_output, 'bi_transfer_iter_{}'.format(it))

                pairwise = tf.add(ang_output, bi_output,
                                  name='pairwise_term_at_iter_{}'.format(it))

                outputs = tf.add(unary, pairwise,
                                 name='energy_at_iter_{}'.format(it))

                inputs = outputs

        return outputs

    def _locally_connected_layer(
            self, layer_name, inputs, bilateral_filters,
            angular_filters, bi_angular_filters, condensing_kernel, sizes=[
                3, 5],
            padding='SAME'):
        """Locally connected layer with non-trainable filter parameters)

        Args:
          layer_name: layer name
          inputs: input tensor with shape 
              [batch_size, zenith, azimuth, num_class].
          bilateral_filters: bilateral filter weight with shape 
              [batch_size, zenith, azimuth, sizes[0]*size[1]-1].
          angular_filters: angular filter weight with shape 
              [sizes[0], sizes[1], in_channel, in_channel].
          condensing_kernel: tensor with shape 
              [size[0], size[1], num_class, (sizes[0]*size[1]-1)*num_class]
          sizes: size of the local region to be filtered.
          padding: padding strategy
        Returns:
          ang_output: output tensor filtered by anguler filter with shape 
              [batch_size, zenith, azimuth, num_class].
          bi_output: output tensor filtered by bilateral filter with shape 
              [batch_size, zenith, azimuth, num_class].
        """
        assert padding == 'SAME', 'only support SAME padding strategy'
        assert sizes[0] % 2 == 1 and sizes[1] % 2 == 1, \
            'Currently only support odd filter size.'

        size_z, size_a = sizes
        pad_z, pad_a = size_z//2, size_a//2
        half_filter_dim = (size_z*size_a)//2
        batch, zenith, azimuth, in_channel = inputs.shape.as_list()

        with tf.variable_scope(layer_name) as scope:
            # message passing
            ang_output = tf.nn.conv2d(
                inputs, angular_filters, [1, 1, 1, 1], padding=padding,
                name='angular_filtered_term'
            )

            bi_ang_output = tf.nn.conv2d(
                inputs, bi_angular_filters, [1, 1, 1, 1], padding=padding,
                name='bi_angular_filtered_term'
            )
            condensed_input = tf.reshape(
                tf.nn.conv2d(
                    inputs*self.lidar_mask, condensing_kernel, [1, 1, 1, 1], padding=padding,
                    name='condensed_prob_map'
                ),
                [batch, zenith, azimuth, size_z*size_a-1, in_channel]
            )

            bi_output = tf.multiply(
                tf.reduce_sum(condensed_input*bilateral_filters, axis=3),
                self.lidar_mask,
                name='bilateral_filtered_term'
            )
            bi_output *= bi_ang_output

        return ang_output, bi_output

    def _bilateral_filter_layer(
            self, layer_name, inputs, thetas=[0.9, 0.01], sizes=[3, 5], stride=1,
            padding='SAME'):
        """Computing pairwise energy with a bilateral filter for CRF.

        Args:
          layer_name: layer name
          inputs: input tensor with shape [batch_size, zenith, azimuth, 2] where the
              last 2 elements are intensity and range of a lidar point.
          thetas: theta parameter for bilateral filter.
          sizes: filter size for zenith and azimuth dimension.
          strides: kernel strides.
          padding: padding.
        Returns:
          out: bilateral filter weight output with size
              [batch_size, zenith, azimuth, sizes[0]*sizes[1]-1, num_class]. Each
              [b, z, a, :, cls] represents filter weights around the center position
              for each class.
        """

        assert padding == 'SAME', 'currently only supports "SAME" padding stategy'
        assert stride == 1, 'currently only supports striding of 1'
        assert sizes[0] % 2 == 1 and sizes[1] % 2 == 1, \
            'Currently only support odd filter size.'

        theta_a, theta_r = thetas
        size_z, size_a = sizes
        pad_z, pad_a = size_z//2, size_a//2
        half_filter_dim = (size_z*size_a)//2
        batch, zenith, azimuth, in_channel = inputs.shape.as_list()

        # assert in_channel == 1, 'Only support input channel == 1'

        with tf.variable_scope(layer_name) as scope:
            condensing_kernel = tf.constant(
                utils.condensing_matrix(size_z, size_a, in_channel),
                dtype=tf.float32,
                name='condensing_kernel'
            )

            condensed_input = tf.nn.conv2d(
                inputs, condensing_kernel, [1, 1, stride, 1], padding=padding,
                name='condensed_input'
            )

            # diff_intensity = tf.reshape(
            #     inputs[:, :, :], [batch, zenith, azimuth, 1]) \
            #     - condensed_input[:, :, :, ::in_channel]

            # Autodetect batch size in graph definition.
            batch = -1

            diff_x = tf.reshape(
                inputs[:, :, :, 0], [batch, zenith, azimuth, 1]) \
                - condensed_input[:, :, :, 0::in_channel]
            diff_y = tf.reshape(
                inputs[:, :, :, 1], [batch, zenith, azimuth, 1]) \
                - condensed_input[:, :, :, 1::in_channel]
            diff_z = tf.reshape(
                inputs[:, :, :, 2], [batch, zenith, azimuth, 1]) \
                - condensed_input[:, :, :, 2::in_channel]

            bi_filters = []
            for cls in range(self.NUM_CLASS):
                theta_a = self.BILATERAL_THETA_A[cls]
                theta_r = self.BILATERAL_THETA_R[cls]
                bi_filter = tf.exp(-(diff_x ** 2 + diff_y ** 2 +
                                     diff_z ** 2) / 2 / theta_r ** 2)
                bi_filters.append(bi_filter)
            out = tf.transpose(
                tf.stack(bi_filters),
                [1, 2, 3, 4, 0],
                name='bilateral_filter_weights'
            )

        return out

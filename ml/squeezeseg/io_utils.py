import logging
import numpy as np
import tensorflow as tf 

from ml.squeezeseg.io import KittiSqueezeSegIO

_logger = logging.getLogger('SqueezeSeg::io_utils')

def _group_numpy_fns(fn_list):
    # example filename: "2011_09_26_0091_0000000062.npy"
    #                          date_ idx_     frame.npy
    groups = defaultdict(list)
    for fn in fn_list:
        group_idx = '_'.join(fn.split('_')[:3])
        group[group_idx].append(fn)
    return groups

def _create_composite_numpy(fn_list):
    X_list = []
    Y_list = []
    for fn in fn_list:
        matrix = np.load(fn)
        X_list.append(matrix[:, :, :KittiSqueezeSegIO.LABEL_TAG])
        Y_list.append(matrix[:, :, KittiSqueezeSegIO.LABEL_TAG])

    composite_X = np.stack(X_list, axis=-1)
    composite_Y = np.stack(Y_list, axis=-1)
    return composite_X, composite_Y

def np_to_tfrecords(X, file_path, verbose=False):
    """ Taken from https://gist.github.com/swyoon/8185b3dcf08ec728fb22b99016dd533f.
        Author: Sangwoong Yoon

    Converts a Numpy array (or two Numpy arrays) into a tfrecord file.
    For supervised learning, feed training inputs to X and training labels to Y.
    For unsupervised learning, only feed training inputs to X, and feed None to Y.
    The length of the first dimensions of X and Y should be the number of samples.

    Parameters
    ----------
    X : numpy.ndarray of rank 2
        Numpy array for training inputs. Its dtype should be float32, float64, or int64.
        If X has a higher rank, it should be rshape before fed to this function.
    Y : numpy.ndarray of rank 2 or None
        Numpy array for training labels. Its dtype should be float32, float64, or int64.
        None if there is no label array.
    file_path : str
        The path and name of the resulting tfrecord file to be generated, with '.tfrecords'
    verbose : bool
        If true, progress is reported.

    Raises
    ------
    ValueError
        If input type is not float (64 or 32) or int.

    """
    def _dtype_feature(ndarray):
        """match appropriate tf.train.Feature class with dtype of ndarray. """
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
        elif dtype_ == np.int64:
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
        else:
            raise ValueError("The input should be numpy ndarray. \
                              Instaed got {}".format(ndarray.dtype))

    assert isinstance(X, np.ndarray)

    # Load appropriate tf.train.Feature class depending on dtype.
    dtype_feature_x = _dtype_feature(X)

    # Generate tfrecord writer.
    writer = tf.python_io.TFRecordWriter(file_path)
    if verbose:
        msg = "Serializing {:d} examples into {}".format(X.shape[0], result_tf_file)
        _logger.info(msg)

    lidar_mask = np.reshape(
        (X[:, :, KittiSqueezeSegIO.DEPTH_TAG] > 0), 
        (self.zenith_level, self.azimuth_level, 1)
    )
    lidar_input = np.reshape(
        X[:, :, :KittiSqueezeSegIO.LABEL_TAG], 
        (self.zenith_level, self.azimuth_level, 1)
    )
    label = X[:, :, :KittiSqueezeSegIO.LABEL_TAG]
    weight = np.zeros(label.shape)
    for l in range(KittiSqueezeSegIO.NUM_CLASS):
        weight[label==l] = KittiSqueezeSegIO.CLS_LOSS_WEIGHT[int(l)]
    d_feature = {
        'lidar_mask': dtype_feature_x(lidar_mask.ravel()),
        'lidar_input': dtype_feature_x(lidar_input.ravel()),
        'label': dtype_feature_x(label.ravel()),
        'weight': dtype_feature_x(weight.ravel()),
    }

    features = tf.train.Features(feature=d_feature)
    example = tf.train.Example(features=features)
    serialized = example.SerializeToString()
    writer.write(serialized)
    writer.close()

    if verbose:
        _logger.info("Writing {} done!".format(result_tf_file))

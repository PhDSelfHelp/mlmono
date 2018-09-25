import tensorflow as tf
from tf.data import Dataset

from ml.base.io_utils import find_tfrecords_in_dir, create_dir_if_not_exist


class MLIO(object):
    # DEFAULT IO Hyperparmaeters listed here as constants:
    # INTERLEAVE_BLOCK -> how many continous data from one single file.
    # INTERLEAVE_CYCLE -> how many concurrent open files.
    # See https://www.tensorflow.org/performance/datasets_performance.

    INTERLEAVE_BLOCK = 1
    INTERLEAVE_CYCLE = 100

    DATA_FILE_PATTERN = "*.tfrecord"

    def __init__(self, global_config):
        self.global_config = global_config
        self.io_config = global_config.io

        self.dataset = None
        self.iterator = None
        self.summary_writer = None

        # Data reading configs for tf.data.TFRecordDataset.
        self.data_enable_download = getattr(self.io_config,
                                            'data_enable_download', False)
        self.batch_size = self.io_config.batch_size
        self.interleave_block = getattr(self.io_config, 'interleave_block',
                                        INTERLEAVE_BLOCK)
        self.interleave_cycle = getattr(self.io_config, 'interleave_cycle',
                                        INTERLEAVE_CYCLE)
        self.data_file_pattern = getattr(self.io_config, 'data_file_pattern',
                                         DATA_FILE_PATTERN)
        self.data_dir = getattr(self.io_config, 'data_dir', None)

        self.filenames = getattr(self.io_config, 'filenames', None)
        if not self.filenames:
            if not self.data_dir:
                msg = "data_dir and filenames must have one and only one defined in config::io."
                raise ValueError(msg)
            self.filenames = find_tfrecords_in_dir(data_dir, self.data_dir)

        # Logs and summary saving configs for tf.summary.FileWriter.
        self.logs_dir = self.io_config.logs_dir
        self.logs_flush_secs = getattr(self.io_config, 'logs_flush_secs', 120)

        # Model saving configs for tf.estimator checkpoints.
        self.model_dir = io_config.model_dir

        # Generate dirs.
        create_dirs_if_not_exist(self.model_dir)
        create_dirs_if_not_exist(self.logs_dir)

        # Generate tf dataset.
        if data_enable_download:
            self.download_data_if_not_exist(self.data_dir)
        self.dataset = self._gen_tf_dataset()

    @classmethod
    def from_config(cls, global_config):
        io_config = global_config.io
        subcls = find_subclass_by_name(cls, io_config.io_name)
        return subcls.from_config(global_config)

    def _gen_tf_dataset(self):
        raise NotImplementedError

    def gen_input_fn(self, num_epochs):
        raise NotImplementedError

    @staticmethod
    def parse_file(filename):
        raise NotImplementedError

    @staticmethod
    def download_data_if_not_exist(data_dir):
        raise NotImplementedError


class TFRecordIO(MLIO):

    @classmethod
    def from_config(cls, io_config):
        io = cls(io_config)
        io.summary_writer = tf.summary.FileWriter(
            self.logs_dir, graph=tf.get_default_graph())
        return io

    def gen_input_fn(self, num_epochs):

        def input_fn():
            self.iterator = self.dataset.make_one_shot_iterator()

            data_ite = iterator.get_next()
            features, labels = self.parse_data(data_ite)
            return features, labels

        return input_fn

    def _gen_tf_dataset(self):
        list_files = Dataset.list_files(self.filenames)
        list_files = list_files.shuffle(self.io_config.fn_shuffle_buffer)

        # Parallel_interleave is preferred as it's deterministic in ordering,
        # this ensures better reproducibility.
        dataset = list_files.apply(
            tf.contrib.data.parallel_interleave(
                lambda filename: self.parse_file(filename),
                cycle_length=self.interleave_cycle,
                block_length=self.interleave_block))

        # The data_shuffle_buffer should be some value > rows in single data shard (record).
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.shuffle(self.io_config.data_shuffle_buffer)
        dataset = dataset.repeat(num_epochs)
        return dataset

    @staticmethod
    def parse_file(filename):
        raise NotImplementedError
        return data_chunk

    @staticmethod
    def parse_data(data_chunk):
        raise NotImplementedError
        return features, labels

    @staticmethod
    def download_data_if_not_exist(data_dir):
        raise NotImplementedError


class KerasDatasetIO(MLIO):

    @classmethod
    def from_config(cls, io_config):
        io = cls(io_config)
        io.summary_writer = tf.summary.FileWriter(
            self.logs_dir, graph=tf.get_default_graph())
        io.dataset = _gen_tf_dataset(load_data_func)
        return io

    def gen_input_fn(self, num_epochs):

        def input_fn():
            x = np.arange(4).reshape(-1, 1).astype('float32')
            ds_x = Dataset.from_tensor_slices(x).repeat().batch(self.batch_size)
            it_x = ds_x.make_one_shot_iterator()

            y = np.arange(5, 9).reshape(-1, 1).astype('float32')
            ds_y = Dataset.from_tensor_slices(y).repeat().batch(self.batch_size)
            it_y = ds_y.make_one_shot_iterator()

            iterator = dataset.make_one_shot_iterator()
            data_ite = iterator.get_next()
            features, labels = self.parse_data(data_ite)
            return features, labels

        return input_fn

    def _gen_tf_dataset(keras_dataset):
        '''Construct a data generator using `tf.Dataset`. '''

        def map_fn(image, label):
            '''Preprocess raw data to trainable input. '''
            x = tf.reshape(tf.cast(image, tf.float32), (28, 28, 1))
            y = tf.one_hot(tf.cast(label, tf.uint8), _NUM_CLASSES)
            return x, y

        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        mode = self.global_config.mode
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if is_training:
            dataset = dataset.shuffle(self.data_shuffle_buffer)
        dataset = dataset.map(map_fn)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

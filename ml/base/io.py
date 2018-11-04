import tensorflow as tf

from ml.base.base_utils import find_subclass_by_name
from ml.base.io_utils import find_tfrecords_in_dir, create_dir_if_not_exist
from ml.base.trainer_utils import is_training


class MLIO(object):
    # DEFAULT IO Hyperparmaeters listed here as constants:
    # INTERLEAVE_BLOCK -> how many continous data from one single file.
    # INTERLEAVE_CYCLE -> how many concurrent open files.
    # See https://www.tensorflow.org/performance/datasets_performance.

    DATA_FILE_PATTERN = "*.tfrecord"

    INTERLEAVE_BLOCK = 1
    INTERLEAVE_CYCLE = 100
    DATA_SHUFFLE_BUFFER = 128
    NUM_PARALLEL_PARSE = 4

    def __init__(self, global_config):
        self.global_config = global_config
        self.io_config = self.global_config.io

        self.mode = self.global_config.mode

        self.dataset = None
        self.iterator = None
        self.summary_writer = None

        # Data reading configs for tf.data.TFRecordDataset.
        self.batch_size = self.global_config.trainer.batch_size

        self.data_enable_download = getattr(self.io_config,
                                            'data_enable_download', False)
        self.data_file_pattern = getattr(self.io_config, 'data_file_pattern',
                                         MLIO.DATA_FILE_PATTERN)
        self.interleave_block = getattr(self.io_config, 'interleave_block',
                                        MLIO.INTERLEAVE_BLOCK)
        self.interleave_cycle = getattr(self.io_config, 'interleave_cycle',
                                        MLIO.INTERLEAVE_CYCLE)
        self.data_shuffle_buffer = getattr(self.io_config, 'data_shuffle_buffer',
                                           MLIO.DATA_SHUFFLE_BUFFER)
        self.num_parallel_parse = getattr(self.io_config, 'num_parallel_parse',
                                           MLIO.NUM_PARALLEL_PARSE)

        self.data_dir = getattr(self.io_config, 'data_dir', None)

        self.filenames = getattr(self.io_config, 'filenames', None)

        # KerasDatasetIO doesn't need any filenames or data_dir
        if not isinstance(self, KerasDatasetIO):
            if not self.filenames:
                if not self.data_dir:
                    msg = "data_dir and filenames must have one and only one defined in config::io."
                    raise ValueError(msg)
                self.filenames = find_tfrecords_in_dir(self.data_dir, self.data_file_pattern)

        # Model saving configs for tf.estimator checkpoints.
        self.model_dir = self.io_config.model_dir

        # Logs and summary saving configs for tf.summary.FileWriter.
        self.logs_dir = self.io_config.logs_dir
        self.logs_flush_secs = getattr(self.io_config, 'logs_flush_secs', 120)

        # Generate dirs.
        create_dir_if_not_exist(self.model_dir)
        create_dir_if_not_exist(self.logs_dir)

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
    def from_config(cls, global_config):
        io_config = global_config.io
        io = cls(global_config)
        io.summary_writer = tf.summary.FileWriter(
            io.logs_dir, graph=tf.get_default_graph())
        return io

    def gen_input_fn(self, num_epochs):
        # Generate tf dataset.
        if self.data_enable_download:
            self.download_data_if_not_exist(self.data_dir)
        self.dataset = self._gen_tf_dataset()

        # def input_fn():
        #     self.iterator = self.dataset.make_one_shot_iterator()
        #     data_ite = iterator.get_next()
        #     features, labels = self.parse_iter(data_ite)
        #     return features, labels

        def input_fn():
            return self.dataset

        return input_fn

    def _gen_tf_dataset(self):
        list_files = tf.data.Dataset.list_files(self.filenames)
        if is_training(self.mode):
            list_files = list_files.shuffle(self.io_config.fn_shuffle_buffer)

        # Parallel_interleave is preferred as it's deterministic in ordering,
        # this ensures better reproducibility.
        dataset = list_files.apply(
            tf.contrib.data.parallel_interleave(
                tf.data.TFRecordDataset,
                cycle_length=self.interleave_cycle,
                block_length=self.interleave_block))
        dataset = dataset.map(lambda filename: self.parse_file(filename))

        # The data_shuffle_buffer should be some value > rows in single data shard (record).
        dataset = dataset.batch(batch_size=self.batch_size)
        # dataset = dataset.apply(
        #     tf.contrib.data.parallel_interleave(
        #         lambda filename: self.parse_file(filename),
        #         cycle_length=self.interleave_cycle,
        #         block_length=self.interleave_block))

        if is_training(self.mode):
            dataset = dataset.shuffle(self.io_config.data_shuffle_buffer)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.map(map_func=parse_iter,
                              num_parallel_calls=self.num_parallel_parse)
        return dataset

    @staticmethod
    def parse_file(filename):
        raise NotImplementedError
        return data_chunk

    @staticmethod
    def parse_iter(data_chunk):
        raise NotImplementedError
        return features, labels

    @staticmethod
    def download_data_if_not_exist(data_dir):
        raise NotImplementedError


class KerasDatasetIO(MLIO):

    @classmethod
    def from_config(cls, global_config):
        io_config = global_config.io
        io = cls(global_config)
        io.summary_writer = tf.summary.FileWriter(
            io.logs_dir, graph=tf.get_default_graph())
        io.dataset = io._gen_tf_dataset()
        return io

    def gen_input_fn(self, num_epochs):
        self.dataset = self._gen_tf_dataset()

        def input_fn():
            iterator = self.dataset.make_one_shot_iterator()
            data_ite = iterator.get_next()
            features, labels = self.parse_iter(data_ite)
            return features, labels

        return input_fn

    def _gen_tf_dataset(self):
        '''Construct a tf dataset using `tf.Dataset`. '''
        images, labels = self.keras_load_data(self.mode)
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        if is_training(self.mode):
            dataset = dataset.shuffle(self.data_shuffle_buffer)
        dataset = dataset.map(self.parse_keras_tensor)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

    @staticmethod
    def parse_keras_tensor(features, labels):
        raise NotImplementedError

    @staticmethod
    def keras_load_data(mode):
        raise NotImplementedError

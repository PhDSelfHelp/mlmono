import tensorflow as tf
from tf.data import Dataset

from ml.base.io_utils import find_tfrecords_in_dir


class MLIO(object):
    # IO Hyperparmaeters
    # INTERLEAVE_BLOCK -> how many continous data from one single file.
    # INTERLEAVE_CYCLE -> how many concurrent open files.
    # See https://www.tensorflow.org/performance/datasets_performance.

    INTERLEAVE_BLOCK = 1
    INTERLEAVE_CYCLE = 100

    DATA_FILE_PATTERN = "*.tfrecord"

    def __init__(self, io_config):
        self.io_config = io_config

        self.dataset = None
        self.summary_writer = None

        # Data reading configs for tf.data.TFRecordDataset.
        self.batch_size = self.io_config.batch_size
        self.interleave_block = getattr(self.io_config, 'interleave_block', INTERLEAVE_BLOCK)
        self.interleave_cycle = getattr(self.io_config, 'interleave_cycle', INTERLEAVE_CYCLE)
        self.data_file_pattern = getattr(self.io_config, 'data_file_pattern', DATA_FILE_PATTERN)
        self.data_dir = getattr(self.io_config, 'data_dir', None)

        self.filenames = getattr(self.io_config, 'filenames', None)
        if not self.filenames:
            if not self.data_dir:
                msg = "data_dir and filenames must have one and only one defined in config::io."
                raise ValueError(msg)
            self.filenames = find_tfrecords_in_dir(data_dir, self.data_dir)

        # Model saving configs for tf.estimator checkpoints.
        self.model_dir = self.io_config.model_dir

        # Logs and summary saving configs for tf.summary.FileWriter.
        self.logs_dir = self.io_config.logs_dir
        self.logs_flush_secs = getattr(self.io_config, 'logs_flush_secs', 120)

    @classmethod
    def from_config(cls, io_config):
        subcls = find_subclass_by_name(cls, io_config.io_name)
        return subcls.from_config(io_config)

    def gen_input_fn(self, num_epochs):
        raise NotImplementedError

    @staticmethod
    def parse_file(filename):
        raise NotImplementedError


class DefaultIO(MLIO):
    @classmethod
    def from_config(cls, io_config):
        io = cls(io_config)
        io.summary_writer = tf.summary.FileWriter(self.logs_dir,
                                                  graph=tf.get_default_graph()
                                                 )
        io.data = tf.data.TFRecordDataset(self.filenames,
                                          buffer_size=None,
                                          num_parallel_reads=None,
                                         )
        return io

    def gen_input_fn(self, num_epochs):
        def input_fn():
            list_files = Dataset.list_files(self.filenames)
            list_files = list_files.shuffle(self.io_config.fn_shuffle_buffer)

            # Parallel_interleave is preferred as it's deterministic in ordering,
            # this ensures better reproducibility.
            dataset = list_files.apply(
                tf.contrib.data.parallel_interleave(
                    lambda filename: self.parse_file(filename),
                    cycle_length=INTERLEAVE_CYCLE,
                    block_length=INTERLEAVE_BLOCK
                )
            )

            # The data_shuffle_buffer should be some value > rows in single data shard (record).
            dataset = dataset.batch(batch_size=self.batch_size)
            dataset = dataset.shuffle(self.io_config.data_shuffle_buffer)
            dataset = dataset.repeat(num_epochs)
            iterator = dataset.make_one_shot_iterator()

            data_ite = iterator.get_next()
            features, labels = self.parse_data(data_ite)
            return features, labels
        return input_fn

    @staticmethod
    def parse_file(filename):
        raise NotImplementedError
        return data_chunk

    @staticmethod
    def parse_data(data_chunk):
        raise NotImplementedError
        return features, labels

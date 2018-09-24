import tensorflow as tf


class MLIO(object):
    def __init__(self, io_config):
        self.io_config = io_config

        self.dataset = None
        self.summary_writer = None

        # Data reading configs for tf.data.TFRecordDataset.

        # Model saving configs for tf.estimator checkpoints.
        self.model_dir = io_config.model_dir

        # Logs and summary saving configs for tf.summary.FileWriter.
        self.logs_dir = io_config.logs_dir
        self.logs_flush_secs = 120,
        if getattr(io.config, 'logs_flush_secs', None):
            self.logs_flush_secs = io.config.logs_flush_secs

    @classmethod
    def from_config(cls, io_config):
        subcls = find_subclass_by_name(cls, io_config.io_name)
        return subcls.from_config(io_config)


class DefaultIO(MLIO):
    @classmethod
    def from_config(cls, io_config):
        io = cls(io_config)
        io.summary_writer = tf.summary.FileWriter(logs_dir, graph=tf.get_default_graph())
        io.data = tf.data.TFRecordDataset(self.io_config.filenames,
                                          buffer_size=None,
                                          num_parallel_reads=None,
                                         )
        return io

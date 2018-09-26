import tensorflow as tf


class MLIO(object):

    def __init__(self, io_config):
        self.io_config = io_config

        self.dataset = None
        self.summary_writer = None
        self.model_dir = io_config.model_dir

    @classmethod
    def from_config(cls, io_config):
        subcls = find_subclass_by_name(cls, io_config.io_name)
        return subcls.from_config(io_config)

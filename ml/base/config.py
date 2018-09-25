import munch
import yaml

# add overwriting with double underscore functionality


class MLConfig(object):

    @classmethod
    def from_file(cls, fn):
        '''
        Args:
            fn: filename of a yaml config file.
        '''
        with open(fn, 'r') as file:
            content = yaml.load(file)
        config = munch.munchify(content)
        return MLConfig(config.model, config.trainer, config.evaluator,
                        config.io, config.metric)

    @classmethod
    def from_internal_file(cls, config_name):
        pass

    def __init__(self, model, trainer, evaluator, io, metric):
        self.global_config = self
        self.model = model
        self.trainer = trainer
        self.evaluator = evaluator
        self.io = io
        self.metric = metric

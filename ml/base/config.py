import munch
import yaml

# add overwriting with double underscore functionality


class MLConfig(object):

    def __getattr__(self, attr):
        '''
            The __getattr__ magic method is only called when the attribute doesn't exist
            on the instance / class / parent classes. We use it to pass attribute access
            of the MLConfig object directly to its dictionary / munch representation.
        '''
        return getattr(self._internal, attr)

    def __init__(self, internal_munch):
        self._internal = internal_munch

    @classmethod
    def from_dict(cls, dic):
        '''
        Args:
            dict: dictionary of a config.
        '''
        config = munch.munchify(dic)
        return MLConfig(config)


class GlobalConfig(MLConfig):

    def __init__(self, global_config, graph, trainer, predictor, io, metric):
        self._internal = global_config
        self.global_config = global_config
        self.graph = graph
        self.trainer = trainer
        self.predictor = predictor
        self.io = io
        self.metric = metric

    @classmethod
    def from_internal_file(cls, config_name):
        # TODO(jdaaph): Add automatically path finding from base dir.
        raise NotImplementedError

    @classmethod
    def from_file(cls, fn):
        '''
        Args:
            fn: filename of a yaml config file.
        '''
        with open(fn, 'r') as file:
            content = yaml.load(file)
        return cls.from_dict(content)

    @classmethod
    def from_dict(cls, dic):
        '''
        Args:
            dict: dictionary of a config.
        '''
        global_config = munch.munchify(dic)
        component_config_lst = [global_config]

        for component in ['graph', 'trainer', 'predictor', 'io', 'metric']:
            config = munch.munchify(getattr(global_config, component))
            component_config_lst.append(config)

        return GlobalConfig(*component_config_lst)

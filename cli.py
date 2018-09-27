import click
import os
from ml.base import MLConfig, MLEstimator

_CONFIG_PATH = os.path.join(os.getcwd(), 'configs/')

@click.command()
@click.argument('action')
@click.option('--config', help='The config file')
def main(action, config):
    settings = MLConfig.from_file(os.path.join(_CONFIG_PATH, config))
    # The command line action should override the one in config
    settings.global_config.mode = action
    model = MLEstimator.from_config(settings)
    action_case = {
        'train': model.train_fn,
        'predict': model.predict_fn
    }
    try:
        action_case[action]()
    except KeyError:
        MSG = 'The ACTION argument should be among'
        for key in action_case.keys():
            MSG = MSG + " '" + key + "' "
        print(MSG)

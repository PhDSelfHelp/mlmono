import click
import os
from ml.base import MLConfig, MLEstimator

_CONFIG_PATH = os.path.join(os.getcwd(), 'configs/')

@click.command()
@click.argument('action')
@click.option('--config', help='The config file')
def main(action, config):
    settings = MLConfig.from_file(os.path.join(_CONFIG_PATH, config))
    model = MLEstimator.from_config(settings)
    def predict():
        pass
    def train():
        pass
    action_case = {
        'train': train,
        'predict': predict
    }
    try:
        action_case[action]()
    except KeyError:
        MSG = 'The ACTION argument should be among'
        for key in action_case.keys():
            MSG + " '" + key + "' "
        print(MSG)

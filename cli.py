import click

from ml.base import MLConfig
from ml.base.estimator import MLEstimator


@click.group()
def main():
    pass

@click.command()
@click.option('--config', help='The config file', required=True)
def train(config):
    config = MLConfig.from_file(config)
    config.global_config.mode = 'train'
    # TODO(jxwulittlebeans): The command line action should override the one in config.
    estimator = MLEstimator.from_config(config)
    estimator.train()

@click.command()
@click.option('--config', help='The config file', required=True)
def predict(config):
    config = MLConfig.from_file(config)
    config.global_config.mode = 'predict'
    estimator = MLEstimator.from_config(config)
    estimator.predict()

main.add_command(train)
main.add_command(predict)

if __name__ == "__main__":
    main()

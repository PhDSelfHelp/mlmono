import click

import ml
from ml.base import GlobalConfig
from ml.base.estimator import MLEstimator
from ml.squeezeseg.cli import npy2tfrecord


@click.group()
def main():
    pass

@click.command()
@click.option('--config', help='The config file', required=True)
def train(config):
    config = GlobalConfig.from_file(config)
    config.global_config.mode = 'train'
    # TODO(jxwulittlebeans): The command line action should override the one in config.
    estimator = MLEstimator.from_config(config)
    estimator.train()

@click.command()
@click.option('--config', help='The config file', required=True)
def predict(config):
    config = GlobalConfig.from_file(config)
    config.global_config.mode = 'predict'
    estimator = MLEstimator.from_config(config)
    estimator.predict()

main.add_command(train)
main.add_command(predict)

main.add_command(npy2tfrecord)

if __name__ == "__main__":
    main()

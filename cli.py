import click
from ml.base import MLConfig
from ml.base.estimator import MLEstimator

@click.group()
def main():
    pass

@click.command()
@click.option('--config', help='The config file')
def train(config):
    settings = MLConfig.from_file(config)
    # The command line action should override the one in config
    settings.global_config.mode = 'train'
    model = MLEstimator.from_config(settings)
    model.train_fn()

@click.command()
@click.option('--config', help='The config file')
def predict(config):
    settings = MLConfig.from_file(config)
    # The command line action should override the one in config
    settings.global_config.mode = 'predict'
    model = MLEstimator.from_config(settings)
    model.predict_fn()

main.add_command(train)
main.add_command(predict)

if __name__ == "__main__":
    main()

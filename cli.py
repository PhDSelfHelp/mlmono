import click
from ml.base import MLConfig
from ml.base.estimator import MLEstimator

@click.command()
@click.argument('action')
@click.option('--config', help='The config file')
def main(action, config):
    settings = MLConfig.from_file(config)
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
        msg = 'The ACTION argument should be among'
        for key in action_case:
            msg = msg + " '" + key + "' "
        raise(ValueError(msg))

if __name__ == "__main__":
    main()

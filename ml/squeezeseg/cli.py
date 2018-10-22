import click

from ml.base import MLConfig, MLIO


@click.command()
@click.option('--config', help='The config file path.') 
def npy2tfrecord(config):
    config = MLConfig.from_file(config)
    
    # TODO(jxwulittlebean): The command line flags should override config here.
    io = MLIO.from_config(config)

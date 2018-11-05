import click

from ml.base import GlobalConfig
from ml.squeezeseg.io import KittiSqueezeSegIO


@click.command()
@click.option('--config',
              required=True,
              help='The config file path.') 
def npy2tfrecord(config):
    config = GlobalConfig.from_file(config)

    # TODO(jxwulittlebean): The command line flags should override config here.
    io = KittiSqueezeSegIO.from_config(config)
    io.create_tf_record()

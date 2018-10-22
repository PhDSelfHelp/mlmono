import click

from ml.squeezeseg.cli import npy2tfrecord


@click.group()
def main():
    pass


main.add_command(npy2tfrecord)

if __name__ == "__main__":
    main()

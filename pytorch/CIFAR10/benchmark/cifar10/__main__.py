import click

from benchmark.cifar10.train import train
from benchmark.cifar10.infer import infer


@click.group()
def cli():
    pass


cli.add_command(train, name='train')
cli.add_command(infer, name='infer')


if __name__ == '__main__':
    cli()

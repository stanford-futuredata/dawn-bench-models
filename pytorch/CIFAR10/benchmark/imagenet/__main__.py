import click

from benchmark.imagenet.train import train


@click.group()
def cli():
    pass


cli.add_command(train, name='train')

if __name__ == '__main__':
    cli()

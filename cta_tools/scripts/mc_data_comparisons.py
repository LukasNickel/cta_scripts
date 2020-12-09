import click
from aict_tools.io import read_data


@click.command()
@click.argument('pattern')
@click.option('cut_file')
@click.option('output')
def main(pattern, cut_file, output):

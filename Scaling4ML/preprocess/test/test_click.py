import click


@click.command()
@click.option('--start')
@click.option('--end')
def convert(start, end):
    click.echo(type(int(start)))
    click.echo(end)


if __name__ == '__main__':
    convert()

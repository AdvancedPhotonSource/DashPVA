import click
import subprocess

@click.group()
def cli():
    pass

@cli.command()
def hkl3d():
    click.echo('Running HKL 3D Viewer')
    subprocess.run(['python', 'viewer/hkl_3d_viewer.py'])

@cli.command()
def detector():
    click.echo('Running Area Detector Viewer')
    subprocess.run(['python', 'viewer/area_det_viewer.py'])
    
if __name__ == '__main__':
    cli()
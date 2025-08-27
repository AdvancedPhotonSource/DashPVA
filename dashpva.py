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
def slice3d():
    click.echo('Running HKL 3D Slicer -- Standalone')
    subprocess.run(['python', 'viewer/hkl_3d_slice_window.py'])

@cli.command()
def detector():
    click.echo('Running Area Detector Viewer')
    subprocess.run(['python', 'viewer/area_det_viewer.py'])
     
@cli.command()
def sim():
    click.echo('Running Simulator Setup')
    subprocess.run(['python', 'pva_setup/pva_workflow_setup_dialog.py'])
    
if __name__ == '__main__':
    cli()
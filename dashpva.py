import click
import subprocess

@click.group()
def cli():
    pass

@cli.command()
def hkl3d():
    """Launch HKL 3D Slicer - Interactive 3D visualization with real-time slicing
    """
    click.echo('Running HKL 3D Viewer')
    subprocess.run(['python', 'viewer/hkl_3d_viewer.py'])

@cli.command()
def slice3d():
    """(Standalone Mode) Launch HKL 3D Slicer - Interactive 3D visualization with real-time slicing
    """
    click.echo('Running HKL 3D Slicer -- Standalone')
    subprocess.run(['python', 'viewer/hkl_3d_slice_window.py'])

@cli.command()
def detector():
    click.echo('Running Area Detector Viewer')
    subprocess.run(['python', 'viewer/area_det_viewer.py'])
     
@cli.command()
@click.option('--sim', is_flag=True, help='Run the simulator setup instead of the standard setup.')
def setup(sim):
    """Sets up the PVA workflow or the simulator."""
    # if sim:
    #     command = ['python', 'consumers/sim_rsm_data.py']
    #     click.echo('Running simulator setup...')
    #     subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    click.echo('Running standard PVA setup...')
    subprocess.run(['python', 'pva_setup/pva_workflow_setup_dialog.py'])
    
if __name__ == '__main__':
    cli()
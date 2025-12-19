# Register HDF5 compression filters globally (e.g., blosc/bitshuffle)
# Importing hdf5plugin once enables decompression support for all h5py reads in the app.
try:
    import hdf5plugin  # noqa: F401
except Exception:
    pass

import sys
import click
import subprocess

@click.group()
def cli():
    pass

@cli.command()
def run():
    """Open DashPVA launcher menu with process tracking and indicators."""
    click.echo('Opening DashPVA Launcher')
    subprocess.run([sys.executable, 'viewer/launcher.py'])

@cli.command()
def hkl3d():
    """Launch HKL 3D Viewer"""
    click.echo('Running HKL 3D Viewer')
    subprocess.run([sys.executable, 'viewer/hkl_3d_viewer.py'])


@cli.command()
def slice3d():
    """(Standalone Mode) Launch HKL 3D Slicer"""
    click.echo('Running HKL 3D Slicer -- Standalone')
    subprocess.run([sys.executable, 'viewer/hkl_3d_slice_window.py'])


@cli.command()
def detector():
    """Launch Area Detector Viewer"""
    click.echo('Running Area Detector Viewer')
    subprocess.run([sys.executable, 'viewer/area_det_viewer.py'])


@cli.command()
@click.option('--ioc', is_flag=True, help='Run the simulator setup instead of the standard setup.')
def setup(ioc):
    """Sets up the PVA workflow or the simulator."""
    if ioc:
        command = [sys.executable, 'consumers/sim_rsm_data.py']
        click.echo('Running simulator setup...')
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    click.echo('Running standard PVA setup...')
    subprocess.run([sys.executable, 'pva_setup/pva_workflow_setup_dialog.py'])

@cli.command()
def workbench():
    """Launch Workbench - Data Analysis Tool"""
    click.echo('Running Workbench - Data Analysis Tool')
    subprocess.run([sys.executable, 'viewer/workbench/workbench.py'])


@cli.command()
@click.argument('name', type=click.Choice(['scan']))
@click.option('--channel', default='', help='PVA channel (optional).')
@click.option('--config', 'config_path', default='', help='Path to TOML config file (optional).')
def view(name, channel, config_path):
    """Open a specific viewer by name. Currently supported: scan."""
    click.echo(f'Opening view: {name}')
    if name == 'scan':
        command = [sys.executable, 'viewer/scan_view.py']
    else:
        raise click.BadParameter(f'Unknown view name: {name}')
    if config_path:
        command.extend(['--config', config_path])
    if channel:
        command.extend(['--channel', channel])
    subprocess.run(command)


if __name__ == '__main__':
    cli()

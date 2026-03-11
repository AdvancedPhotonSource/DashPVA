# Register HDF5 compression filters globally (e.g., blosc/bitshuffle)
# Importing hdf5plugin once enables decompression support for all h5py reads in the app.
try:
    import hdf5plugin  # noqa: F401
except Exception:
    pass

import sys
import click
import subprocess

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
@click.option('-d', '--detector', 'help_detector', is_flag=True, help='Show help for detector')
@click.option('-r', '--run', 'help_run', is_flag=True, help='Show help for run')
@click.option('-k', '--hkl3d', 'help_hkl3d', is_flag=True, help='Show help for hkl3d')
@click.option('-l', '--slice3d', 'help_slice3d', is_flag=True, help='Show help for slice3d')
@click.option('-S', '--setup', 'help_setup', is_flag=True, help='Show help for setup')
@click.option('-w', '--workbench', 'help_workbench', is_flag=True, help='Show help for workbench')
@click.option('-m', '--monitor', 'help_monitor', is_flag=True, help='Show help for monitor')
@click.pass_context
def cli(ctx, help_detector, help_run, help_hkl3d, help_slice3d, help_setup, help_workbench, help_monitor):
    """
    DashPVA: High-Performance X-ray Visualization & Analysis Tool.
    
    This suite provides real-time monitoring (PVA), 3D Reciprocal Space Mapping (HKL), 
    and post-processing workbenches to analyze and manipulate your data.
    """
    # Handle global help flags that print subcommand help and exit
    selected = [name for name, flag in [
        ('detector', help_detector),
        ('run', help_run),
        ('hkl3d', help_hkl3d),
        ('slice3d', help_slice3d),
        ('setup', help_setup),
        ('workbench', help_workbench),
        ('monitor', help_monitor),
    ] if flag]

    if len(selected) > 1:
        raise click.UsageError('Please pick only one global help flag (e.g., -d/--detector, -r/--run, -k/--hkl3d, -l/--slice3d, -S/--setup, -w/--workbench, -v/--view).')

    if len(selected) == 1:
        sub_name = selected[0]
        sub_cmd = cli.get_command(ctx, sub_name)
        if sub_cmd is None:
            raise click.UsageError(f'Unknown command: {sub_name}')
        sub_ctx = click.Context(sub_cmd, info_name=f"{ctx.info_name} {sub_name}", parent=ctx)
        click.echo(sub_cmd.get_help(sub_ctx))
        ctx.exit()

    # Otherwise, proceed normally

@cli.command()
def run():
    """Open DashPVA launcher menu with process tracking and indicators."""
    click.echo('Opening DashPVA Launcher')
    # Use module entrypoint for reliable relative imports and the registry-based launcher
    exit_code = subprocess.run([sys.executable, '-m', 'viewer.launcher.launcher']).returncode
    sys.exit(exit_code)

@cli.command()
def hkl3d():
    """Launch HKL 3D Viewer"""
    click.echo('Running HKL 3D Viewer')
    exit_code = subprocess.run([sys.executable, 'viewer/hkl_3d_viewer.py']).returncode
    sys.exit(exit_code)


@cli.command()
def slice3d():
    """(Standalone Mode) Launch HKL 3D Slicer"""
    click.echo('Running HKL 3D Slicer -- Standalone')
    exit_code = subprocess.run([sys.executable, 'viewer/hkl_3d_slice_window.py']).returncode
    sys.exit(exit_code)


@cli.command()
def detector():
    """Launch Area Detector Viewer"""
    click.echo('Running Area Detector Viewer')
    exit_code = subprocess.run([sys.executable, 'viewer/area_det_viewer.py']).returncode
    sys.exit(exit_code)


@cli.command()
@click.option('--ioc', is_flag=True, help='Run the simulator setup instead of the standard setup.')
def setup(ioc):
    """Sets up the PVA workflow or the simulator."""
    if ioc:
        click.echo('Running simulator setup...')
        subprocess.Popen([sys.executable, 'consumers/sim_rsm_data.py'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    click.echo('Running standard PVA setup...')
    exit_code = subprocess.run([sys.executable, 'pva_setup/pva_workflow_setup_dialog.py']).returncode
    sys.exit(exit_code)

@cli.command()
def workbench():
    """Launch Workbench - Data Analysis Tool"""
    click.echo('Running Workbench - Data Analysis Tool')
    exit_code = subprocess.run([sys.executable, 'viewer/workbench/workbench.py']).returncode
    sys.exit(exit_code)


@cli.command()
@click.argument('name', type=click.Choice(['scan', 'scan-monitors']))
@click.option('--channel', default='', help='PVA channel (optional).')
@click.option('--config', 'config_path', default='', help='Path to TOML config file (optional).')
def monitor(name, channel, config_path):
    """Open a specific monitor by name. Supported: scan (alias: scan-monitors)."""
    click.echo(f'Opening monitor: {name}')
    if name in ('scan', 'scan-monitors'):
        command = [sys.executable, 'viewer/scan_view.py']
    else:
        raise click.BadParameter(f'Unknown view name: {name}')
    if config_path:
        command.extend(['--config', config_path])
    if channel:
        command.extend(['--channel', channel])
    exit_code = subprocess.run(command).returncode
    sys.exit(exit_code)


if __name__ == '__main__':
    cli()

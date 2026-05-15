import subprocess
import sys

import click

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
@click.option('-d', '--detector', 'help_detector', is_flag=True, help='Show help for detector')
@click.option('-r', '--run', 'help_run', is_flag=True, help='Show help for run')
@click.option('-k', '--hkl3d', 'help_hkl3d', is_flag=True, help='Show help for hkl3d')
@click.option('-S', '--setup', 'help_setup', is_flag=True, help='Show help for setup')
@click.option('-w', '--workbench', 'help_workbench', is_flag=True, help='Show help for workbench')
@click.option('-m', '--monitor', 'help_monitor', is_flag=True, help='Show help for monitor')
@click.pass_context
def cli(ctx, help_detector, help_run, help_hkl3d, help_setup, help_workbench, help_monitor):
    """
    DashPVA: High-Performance X-ray Visualization & Analysis Tool.

    This suite provides real-time monitoring (PVA), 3D Reciprocal Space Mapping (HKL),
    and post-processing workbenches to analyze and manipulate your data.
    """
    selected = [name for name, flag in [
        ('detector', help_detector),
        ('run', help_run),
        ('hkl3d', help_hkl3d),
        ('setup', help_setup),
        ('workbench', help_workbench),
        ('monitor', help_monitor),
    ] if flag]

    if len(selected) > 1:
        raise click.UsageError('Please pick only one global help flag (e.g., -d/--detector, -r/--run, -k/--hkl3d, -S/--setup, -w/--workbench, -m/--monitor).')

    if len(selected) == 1:
        sub_name = selected[0]
        sub_cmd = cli.get_command(ctx, sub_name)
        if sub_cmd is None:
            raise click.UsageError(f'Unknown command: {sub_name}')
        sub_ctx = click.Context(sub_cmd, info_name=f"{ctx.info_name} {sub_name}", parent=ctx)
        click.echo(sub_cmd.get_help(sub_ctx))
        ctx.exit()


@cli.command()
def run():
    """Open DashPVA launcher menu with process tracking and indicators."""
    click.echo('Opening DashPVA Launcher')
    exit_code = subprocess.run([sys.executable, '-m', 'dashpva.viewer.launcher.launcher']).returncode
    sys.exit(exit_code)

@cli.command()
def hkl3d():
    """Launch HKL 3D Viewer"""
    click.echo('Running HKL 3D Viewer')
    exit_code = subprocess.run([sys.executable, '-m', 'dashpva.viewer.hkl3d.hkl_3d_viewer']).returncode
    sys.exit(exit_code)


@cli.command()
def detector():
    """Launch Area Detector Viewer"""
    click.echo('Running Area Detector Viewer')
    exit_code = subprocess.run([sys.executable, '-m', 'dashpva.viewer.area_det_viewer']).returncode
    sys.exit(exit_code)


@cli.command()
@click.option('--ioc', is_flag=True, help='Run the simulator setup instead of the standard setup.')
def setup(ioc):
    """Sets up the PVA workflow or the simulator."""
    if ioc:
        click.echo('Running simulator setup...')
        subprocess.Popen([sys.executable, '-m', 'dashpva.consumers.caIOC_servers.sim_rsm_data'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    click.echo('Running standard PVA setup...')
    exit_code = subprocess.run([sys.executable, '-m', 'dashpva.workflow.workflow']).returncode
    sys.exit(exit_code)

@cli.command()
def bayesian():
    """Launch Bayesian 2-D Scan Viewer"""
    click.echo('Running Bayesian 2-D Scan Viewer')
    subprocess.run([sys.executable, '-m', 'dashpva.viewer.bayesian.bayesian_viewer'])


@cli.command()
def workbench():
    """Launch Workbench - Data Analysis Tool"""
    click.echo('Running Workbench - Data Analysis Tool')
    exit_code = subprocess.run([sys.executable, '-m', 'dashpva.viewer.workbench.workbench']).returncode
    sys.exit(exit_code)


@cli.command()
def h5viewer():
    """Launch HDF5 Viewer — interactive HDF5 file browser and image viewer."""
    click.echo('Running HDF5 Viewer')
    exit_code = subprocess.run([sys.executable, '-m', 'dashpva.hdf_viewer.interactive']).returncode
    sys.exit(exit_code)


@cli.command()
def pyfai():
    """Launch pyFAI 1D Reduction — live azimuthal integration."""
    click.echo('Running pyFAI 1D Reduction')
    exit_code = subprocess.run([sys.executable, '-m', 'dashpva.viewer.pyFAI_analysis']).returncode
    sys.exit(exit_code)


@cli.command()
def phasefitter():
    """Launch XRD Phase Fitter — fit crystal phases to 1D diffraction patterns."""
    click.echo('Running XRD Phase Fitter')
    exit_code = subprocess.run([sys.executable, '-m', 'dashpva.viewer.phase_fitter']).returncode
    sys.exit(exit_code)


@cli.command()
@click.argument('name', type=click.Choice(['scan', 'scan-monitors']))
@click.option('--channel', default='', help='PVA channel (optional).')
def monitor(name, channel):
    """Open a specific monitor by name. Supported: scan (alias: scan-monitors)."""
    click.echo(f'Opening monitor: {name}')
    if name in ('scan', 'scan-monitors'):
        command = [sys.executable, '-m', 'dashpva.viewer.scan_view']
    else:
        raise click.BadParameter(f'Unknown view name: {name}')
    if channel:
        command.extend(['--channel', channel])
    exit_code = subprocess.run(command).returncode
    sys.exit(exit_code)


if __name__ == '__main__':
    cli()

import os
import subprocess
import sys
from pathlib import Path

import click

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
@click.option('-d', '--detector', 'help_detector', is_flag=True, help='Show help for detector')
@click.option('-r', '--run', 'help_run', is_flag=True, help='Show help for run')
@click.option('-k', '--hkl3d', 'help_hkl3d', is_flag=True, help='Show help for hkl3d')
@click.option('-S', '--setup', 'help_setup', is_flag=True, help='Show help for setup')
@click.option('-w', '--workbench', 'help_workbench', is_flag=True, help='Show help for workbench')
@click.option('-m', '--monitor', 'help_monitor', is_flag=True, help='Show help for monitor')
@click.option('-s', '--sim', 'help_sim', is_flag=True, help='Show help for sim')
@click.pass_context
def cli(ctx, help_detector, help_run, help_hkl3d, help_setup, help_workbench, help_monitor, help_sim):
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
        ('sim', help_sim),
    ] if flag]

    if len(selected) > 1:
        raise click.UsageError('Please pick only one global help flag (e.g., -d/--detector, -r/--run, -k/--hkl3d, -S/--setup, -w/--workbench, -m/--monitor, -s/--sim).')

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
@click.option('--pv', 'pv', default=None, metavar='CHANNEL',
              help='Open a full PVA channel directly, used verbatim (e.g. s6lambda1:Pva1:Image).')
@click.option('--prefix', 'prefix', default=None, metavar='PREFIX',
              help='Open a detector prefix directly; ":Pva1:Image" is appended (e.g. s6lambda1).')
def detector(pv, prefix):
    """Launch Area Detector Viewer.

    With no options the prefix dialog is shown. Use --pv to open a full channel
    as-is, or --prefix to append ":Pva1:Image" to a detector prefix.
    """
    if pv and prefix:
        raise click.UsageError('Use only one of --pv or --prefix.')
    click.echo('Running Area Detector Viewer')
    cmd = [sys.executable, '-m', 'dashpva.viewer.area_det.area_det_viewer']
    if pv and pv.strip():
        cmd += ['--channel', pv.strip()]
    elif prefix and prefix.strip():
        cmd += ['--channel', f'{prefix.strip()}:Pva1:Image']
    exit_code = subprocess.run(cmd).returncode
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
    """Launch the Bayesian Optimization viewer (blop)"""
    click.echo('Running Bayesian Optimization viewer (blop)')
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


_SIM_MODULE = 'dashpva.consumers.caIOC_servers.ad_sim_server_modified'
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _build_sim_cmd(*, channel, fps, dt, nf, rt, rp, nx, ny, mpv, input_file):
    cmd = [
        sys.executable, '-u', '-m', _SIM_MODULE,
        '-cn', channel,
        '-fps', str(fps),
        '-dt', dt,
        '-nf', str(nf),
        '-rt', str(rt),
        '-rp', str(rp),
    ]
    if nx is not None:
        cmd.extend(['-nx', str(nx)])
    if ny is not None:
        cmd.extend(['-ny', str(ny)])
    if mpv:
        cmd.extend(['-mpv', mpv])
    if input_file:
        path = input_file if os.path.isabs(input_file) else str(_PROJECT_ROOT / input_file)
        cmd.extend(['-if', path])
    return cmd


def _sim_options(fn):
    """Shared click options for sim presets."""
    opts = [
        click.option('--channel', '-cn', default=None, help='PVA channel name.'),
        click.option('--fps', type=float, default=None, help='Frames per second.'),
        click.option('--dt', default=None, help='Data type (uint8, uint16, float32, ...).'),
        click.option('--nf', type=int, default=None, help='Number of frames.'),
        click.option('--rt', type=float, default=None, help='Runtime in seconds.'),
        click.option('--rp', type=int, default=None, help='Report period.'),
        click.option('--nx', type=int, default=None, help='Width in pixels.'),
        click.option('--ny', type=int, default=None, help='Height in pixels.'),
        click.option('--mpv', default=None, help='Metadata output PVs (comma-separated).'),
        click.option('--input-file', '-if', 'input_file', default=None, help='Input image file.'),
    ]
    for opt in reversed(opts):
        fn = opt(fn)
    return fn


@cli.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@_sim_options
@click.pass_context
def sim(ctx, **kwargs):
    """Run area detector simulation server.

    With no subcommand, starts a random-frame simulation matching the
    PVA Workflow GUI defaults (1024x1024, 10 fps, uint8).

    \b
    Presets:
      DashPVA sim          Random frames (default)
      DashPVA sim pyfai    CeO2 diffraction image for pyFAI
    """
    if ctx.invoked_subcommand is not None:
        return
    defaults = dict(
        channel='pvapy:image', fps=10, dt='uint8', nf=900,
        rt=28800, rp=100, nx=1024, ny=1024, mpv='ca://x,ca://y',
        input_file=None,
    )
    merged = {k: (kwargs[k] if kwargs[k] is not None else defaults[k]) for k in defaults}
    click.echo('Starting area detector simulation (random frames)...')
    cmd = _build_sim_cmd(**merged)
    sys.exit(subprocess.run(cmd).returncode)


@sim.command('pyfai')
@_sim_options
def sim_pyfai(**kwargs):
    """Run simulation with CeO2 diffraction image for pyFAI integration."""
    defaults = dict(
        channel='pvapy:image', fps=10, dt='uint16', nf=20,
        rt=7200, rp=100, nx=None, ny=None, mpv='ca://x,ca://y',
        input_file='tests/test_data/d350_CeO2-000000.tif',
    )
    merged = {k: (kwargs[k] if kwargs[k] is not None else defaults[k]) for k in defaults}
    click.echo('Starting area detector simulation (CeO2 pyFAI)...')
    cmd = _build_sim_cmd(**merged)
    sys.exit(subprocess.run(cmd).returncode)


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

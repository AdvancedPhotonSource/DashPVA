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
@click.option('--ioc', is_flag=True, help='Run the simulator setup instead of the standard setup.')
def setup(ioc):
    """Sets up the PVA workflow or the simulator."""
    if ioc:
        command = ['python', 'consumers/sim_rsm_data.py']
        click.echo('Running simulator setup...')
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    click.echo('Running standard PVA setup...')
    subprocess.run(['python', 'pva_setup/pva_workflow_setup_dialog.py'])
    
@cli.command()
def run():
    """Open DashPVA launcher menu with process tracking and indicators."""
    import sys
    from PyQt5 import uic
    from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox
    from PyQt5.QtCore import QTimer

    class LauncherDialog(QDialog):
        def __init__(self):
            super(LauncherDialog, self).__init__()
            uic.loadUi('gui/dashpva.ui', self)
            self.processes = {}
            self._timer = QTimer(self)
            self._timer.setInterval(500)
            self._timer.timeout.connect(self._poll_processes)
            self._timer.start()

            # Wire buttons to launchers
            if hasattr(self, 'btn_hkl3d_viewer'):
                self.btn_hkl3d_viewer.clicked.connect(
                    lambda: self.launch(
                        'hkl3d_viewer',
                        [sys.executable, 'viewer/hkl_3d_viewer.py'],
                        self.btn_hkl3d_viewer,
                        'HKL 3D Viewer — Running…'
                    )
                )
            if hasattr(self, 'btn_hkl3d_slicer'):
                self.btn_hkl3d_slicer.clicked.connect(
                    lambda: self.launch(
                        'hkl3d_slicer',
                        [sys.executable, 'viewer/hkl_3d_slice_window.py'],
                        self.btn_hkl3d_slicer,
                        'HKL 3D Slicer — Running…'
                    )
                )
            if hasattr(self, 'btn_area_detector'):
                self.btn_area_detector.clicked.connect(
                    lambda: self.launch(
                        'area_detector',
                        [sys.executable, 'viewer/area_det_viewer.py'],
                        self.btn_area_detector,
                        'Area Detector Viewer — Running…'
                    )
                )
            if hasattr(self, 'btn_pva_setup'):
                self.btn_pva_setup.clicked.connect(
                    lambda: self.launch(
                        'pva_setup',
                        [sys.executable, 'pva_setup/pva_workflow_setup_dialog.py'],
                        self.btn_pva_setup,
                        'PVA Workflow Setup — Running…'
                    )
                )
            if hasattr(self, 'btn_sim_setup'):
                self.btn_sim_setup.clicked.connect(
                    lambda: self.launch(
                        'sim_setup',
                        [sys.executable, 'consumers/sim_rsm_data.py'],
                        self.btn_sim_setup,
                        'Simulator — Running…',
                        quiet=True
                    )
                )
            if hasattr(self, 'btn_exit'):
                self.btn_exit.clicked.connect(self.request_close)
            if hasattr(self, 'btn_shutdown_all'):
                self.btn_shutdown_all.clicked.connect(self._confirm_shutdown_all)

            self._update_status()

        def launch(self, key, cmd, button, running_text, quiet=False):
            """Start a child process and update UI indicators."""
            if key in self.processes and self.processes[key]['popen'].poll() is None:
                # Already running
                return
            original_text = button.text()
            button.setEnabled(False)
            button.setText(f"{original_text} — Launching…")

            kwargs = {}
            if quiet:
                import subprocess as _sp
                kwargs['stdout'] = _sp.DEVNULL
                kwargs['stderr'] = _sp.DEVNULL

            try:
                import subprocess as _sp
                p = _sp.Popen(cmd, **kwargs)
                button.setText(running_text)
                self.processes[key] = {
                    'popen': p,
                    'button': button,
                    'original_text': original_text,
                    'running_text': running_text
                }
            except Exception as e:
                QMessageBox.critical(
                    self,
                    'Launch Failed',
                    f'Failed to launch:\n{" ".join(cmd)}\n\n{e}'
                )
                button.setText(original_text)
                button.setEnabled(True)

            self._update_status()

        def _poll_processes(self):
            """Periodic check for finished processes to restore UI state."""
            finished = []
            for key, entry in self.processes.items():
                p = entry['popen']
                if p.poll() is not None:
                    # Process ended
                    entry['button'].setText(entry['original_text'])
                    entry['button'].setEnabled(True)
                    finished.append(key)
            for key in finished:
                self.processes.pop(key, None)
            self._update_status()

        def _update_status(self):
            """Update status label and button states."""
            count = len(self.processes)
            if hasattr(self, 'lbl_status'):
                self.lbl_status.setText('No modules running' if count == 0 else f'{count} module(s) running')
            if hasattr(self, 'btn_exit'):
                # Exit remains enabled; closing will prompt if processes are running
                self.btn_exit.setEnabled(True)
            if hasattr(self, 'btn_shutdown_all'):
                # Enable Shutdown All only when there are running modules
                self.btn_shutdown_all.setEnabled(count > 0)

        def _terminate_proc(self, p, timeout=3.0):
            """Attempt graceful terminate, then force kill if still alive."""
            try:
                if p.poll() is None:
                    p.terminate()
                    try:
                        p.wait(timeout=timeout)
                    except Exception:
                        pass
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass

        def shutdown_all(self):
            """Force-stop all running modules and restore UI state."""
            for key, entry in list(self.processes.items()):
                self._terminate_proc(entry['popen'])
                entry['button'].setText(entry['original_text'])
                entry['button'].setEnabled(True)
                self.processes.pop(key, None)
            self._update_status()

        def _confirm_shutdown_all(self):
            """Confirm and force-stop all running modules."""
            count = len(self.processes)
            if count == 0:
                return
            resp = QMessageBox.question(
                self,
                'Shutdown All Modules',
                'Are you sure you want to force stop all running modules?\n\nData might be lost.',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if resp == QMessageBox.Yes:
                self.shutdown_all()

        def request_close(self):
            """Prompt to force stop modules before exiting if any are running."""
            try:
                any_running = any(entry['popen'].poll() is None for entry in self.processes.values())
            except Exception:
                any_running = False
            if any_running:
                resp = QMessageBox.question(
                    self,
                    'Exit Launcher',
                    'There are running modules.\n\nForce stop all and exit?\n\nData might be lost.',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if resp == QMessageBox.Yes:
                    self.shutdown_all()
                    self.close()
            else:
                self.close()

        def closeEvent(self, event):
            """On close, prompt to force-stop modules if any are running."""
            try:
                any_running = any(entry['popen'].poll() is None for entry in self.processes.values())
            except Exception:
                any_running = False
            if any_running:
                resp = QMessageBox.question(
                    self,
                    'Exit Launcher',
                    'There are running modules.\n\nForce stop all and exit?\n\nData might be lost.',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if resp == QMessageBox.Yes:
                    self.shutdown_all()
                    event.accept()
                else:
                    event.ignore()
            else:
                event.accept()

    app = QApplication(sys.argv)
    dlg = LauncherDialog()
    dlg.show()
    app.exec_()

if __name__ == '__main__':
    cli()

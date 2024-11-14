import sys
import subprocess
import threading
import os
import signal
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QFileDialog, QDialog
from PyQt5.QtCore import pyqtSignal, QObject

class Worker(QObject):
    output_signal = pyqtSignal(str)

    def __init__(self, process):
        super().__init__()
        self.process = process
        self._running = True

    def run(self):
        while self._running:
            output = self.process.stdout.readline()
            if output:
                text = output.strip()
                self.output_signal.emit(text)
            elif self.process.poll() is not None:
                break

    def stop(self):
        self._running = False
        try:
            # Terminate the process group
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        except Exception as e:
            pass

class PVASetupDialog(QDialog):
    def __init__(self, parent=None):
        super(PVASetupDialog, self).__init__(parent)
        uic.loadUi('gui/pva_workflow_setup.ui', self)

        # Initialize process dictionaries
        self.processes = {}
        self.workers = {}

        # Sim Server Tab
        self.buttonBrowse_metadata.clicked.connect(self.browse_metadata_config_sim)
        self.buttonRunSimServer.clicked.connect(self.run_sim_server)
        self.buttonStopSimServer.clicked.connect(self.stop_sim_server)

        # Split Consumers Tab
        self.buttonRunSplitConsumers.clicked.connect(self.run_split_consumers)
        self.buttonStopSplitConsumers.clicked.connect(self.stop_split_consumers)

        # Collector Tab
        self.buttonRunCollector.clicked.connect(self.run_collector)
        self.buttonStopCollector.clicked.connect(self.stop_collector)

        # Analysis Consumer Tab
        self.buttonRunAnalysisConsumer.clicked.connect(self.run_analysis_consumer)
        self.buttonStopAnalysisConsumer.clicked.connect(self.stop_analysis_consumer)

    # Browse functions for Sim Server
    def browse_processor_file_sim(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Sim Server Processor File', '', 'Python Files (*.py)')
        if file_name:
            self.lineEditProcessorFileSim.setText(file_name)

    def browse_metadata_config_sim(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Metadata Config File', '', 'JSON Files (*.json)')
        if file_name:
            self.lineEditMetadataConfigSim.setText(file_name)

    # Browse functions for Split Consumers
    def browse_processor_file_split(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Split Consumers Processor File', '', 'Python Files (*.py)')
        if file_name:
            self.lineEditProcessorFileSplit.setText(file_name)

    def browse_metadata_config_split(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Metadata Config File', '', 'JSON Files (*.json)')
        if file_name:
            self.lineEditMetadataConfigSplit.setText(file_name)

    # Browse functions for Collector
    def browse_processor_file_collector(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Collector Processor File', '', 'Python Files (*.py)')
        if file_name:
            self.lineEditProcessorFileCollector.setText(file_name)

    def browse_metadata_config_collector(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Metadata Config File', '', 'JSON Files (*.json)')
        if file_name:
            self.lineEditMetadataConfigCollector.setText(file_name)

    # Browse functions for Analysis Consumer
    def browse_processor_file_analysis(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Analysis Consumer Processor File', '', 'Python Files (*.py)')
        if file_name:
            self.lineEditProcessorFileAnalysis.setText(file_name)

    def browse_metadata_config_analysis(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Metadata Config File', '', 'JSON Files (*.json)')
        if file_name:
            self.lineEditMetadataConfigAnalysis.setText(file_name)

    # Run and Stop functions for Sim Server
    def run_sim_server(self):
        if 'sim_server' in self.processes:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Sim Server is already running.')
            return

        # Build command
        cmd = [
            'python', '-u', self.lineEditProcessorFileSim.text(),
            '-cn', self.lineEditInputChannelSim.text(),
            '-nx', str(self.spinBoxNx.value()),
            '-ny', str(self.spinBoxNy.value()),
            '-fps', str(self.spinBoxFps.value()),
            '-dt', self.comboBoxDt.currentText(),
            '-nf', str(self.spinBoxNf.value()),
            '-rt', str(self.spinBoxRt.value()),
            '-mpv', self.lineEditMpv.text(),
            '-rp', str(self.spinBoxRp.value())
        ]

        # Add metadata config file if specified
        metadata_config = self.lineEditMetadataConfigSim.text()
        if metadata_config:
            cmd.extend(['--metadata-config', metadata_config])

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                universal_newlines=True
            )
            self.processes['sim_server'] = process

            # Start thread to read output
            worker = Worker(process)
            worker.output_signal.connect(self.textEditSimServerOutput.appendPlainText)
            thread = threading.Thread(target=worker.run)
            thread.daemon = True
            thread.start()
            self.workers['sim_server'] = (worker, thread)

            # Update button states and status label
            self.buttonRunSimServer.setEnabled(False)
            self.buttonStopSimServer.setEnabled(True)
            self.labelStatusSimServer.setText(f'Process ID: {process.pid}')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to start Sim Server: {str(e)}')

    def stop_sim_server(self):
        if 'sim_server' in self.processes:
            self.workers['sim_server'][0].stop()
            self.processes['sim_server'].wait()
            del self.processes['sim_server']
            del self.workers['sim_server']
            self.buttonRunSimServer.setEnabled(True)
            self.buttonStopSimServer.setEnabled(False)
            self.labelStatusSimServer.setText('Process ID: Not running')
            self.textEditSimServerOutput.appendPlainText('Sim Server stopped.')

    # Run and Stop functions for Split Consumers
    def run_split_consumers(self):
        if 'split_consumers' in self.processes:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Split Consumers are already running.')
            return

        cmd = [
            'pvapy-hpc-consumer',
            '--input-channel', self.lineEditInputChannelSplit.text(),
            '--control-channel', self.lineEditControlChannelSplit.text(),
            '--status-channel', self.lineEditStatusChannelSplit.text(),
            '--output-channel', self.lineEditOutputChannelSplit.text(),
            '--processor-file', self.lineEditProcessorFileSplit.text(),
            '--processor-class', self.lineEditProcessorClassSplit.text(),
            '--report-period', str(self.spinBoxReportPeriodSplit.value()),
            '--server-queue-size', str(self.spinBoxServerQueueSizeSplit.value()),
            '--n-consumers', str(self.spinBoxNConsumersSplit.value()),
            '--distributor-updates', str(self.spinBoxDistributorUpdatesSplit.value()),
            '--metadata-channels', self.lineEditMetadataChannelsSplit.text()
        ]

        # Add metadata config file if specified
        metadata_config = self.lineEditMetadataConfigSplit.text()
        if metadata_config:
            cmd.extend(['--metadata-config', metadata_config])

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                universal_newlines=True
            )
            self.processes['split_consumers'] = process

            # Start thread to read output
            worker = Worker(process)
            worker.output_signal.connect(self.textEditSplitConsumersOutput.appendPlainText)
            thread = threading.Thread(target=worker.run)
            thread.daemon = True
            thread.start()
            self.workers['split_consumers'] = (worker, thread)

            # Update button states and status label
            self.buttonRunSplitConsumers.setEnabled(False)
            self.buttonStopSplitConsumers.setEnabled(True)
            self.labelStatusSplitConsumers.setText(f'Process ID: {process.pid}')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to start Split Consumers: {str(e)}')

    def stop_split_consumers(self):
        if 'split_consumers' in self.processes:
            self.workers['split_consumers'][0].stop()
            self.processes['split_consumers'].wait()
            del self.processes['split_consumers']
            del self.workers['split_consumers']
            self.buttonRunSplitConsumers.setEnabled(True)
            self.buttonStopSplitConsumers.setEnabled(False)
            self.labelStatusSplitConsumers.setText('Process ID: Not running')
            self.textEditSplitConsumersOutput.appendPlainText('Split Consumers stopped.')

    # Run and Stop functions for Collector
    def run_collector(self):
        if 'collector' in self.processes:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Collector is already running.')
            return

        cmd = [
            'pvapy-hpc-collector',
            '--collector-id', str(self.spinBoxCollectorId.value()),
            '--producer-id-list', self.lineEditProducerIdList.text(),
            '--input-channel', self.lineEditInputChannelCollector.text(),
            '--control-channel', self.lineEditControlChannelCollector.text(),
            '--status-channel', self.lineEditStatusChannelCollector.text(),
            '--output-channel', self.lineEditOutputChannelCollector.text(),
            '--processor-file', self.lineEditProcessorFileCollector.text(),
            '--processor-class', self.lineEditProcessorClassCollector.text(),
            '--report-period', str(self.spinBoxReportPeriodCollector.value()),
            '--server-queue-size', str(self.spinBoxServerQueueSizeCollector.value()),
            '--collector-cache-size', str(self.spinBoxCollectorCacheSize.value())
        ]

        # Add metadata config file if specified
        metadata_config = self.lineEditMetadataConfigCollector.text()
        if metadata_config:
            cmd.extend(['--metadata-config', metadata_config])

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                universal_newlines=True
            )
            self.processes['collector'] = process

            # Start thread to read output
            worker = Worker(process)
            worker.output_signal.connect(self.textEditCollectorOutput.appendPlainText)
            thread = threading.Thread(target=worker.run)
            thread.daemon = True
            thread.start()
            self.workers['collector'] = (worker, thread)

            # Update button states and status label
            self.buttonRunCollector.setEnabled(False)
            self.buttonStopCollector.setEnabled(True)
            self.labelStatusCollector.setText(f'Process ID: {process.pid}')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to start Collector: {str(e)}')

    def stop_collector(self):
        if 'collector' in self.processes:
            self.workers['collector'][0].stop()
            self.processes['collector'].wait()
            del self.processes['collector']
            del self.workers['collector']
            self.buttonRunCollector.setEnabled(True)
            self.buttonStopCollector.setEnabled(False)
            self.labelStatusCollector.setText('Process ID: Not running')
            self.textEditCollectorOutput.appendPlainText('Collector stopped.')

    # Run and Stop functions for Analysis Consumer
    def run_analysis_consumer(self):
        if 'analysis_consumer' in self.processes:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Analysis Consumer is already running.')
            return

        cmd = [
            'pvapy-hpc-consumer',
            '--input-channel', self.lineEditInputChannelAnalysis.text(),
            '--control-channel', self.lineEditControlChannelAnalysis.text(),
            '--status-channel', self.lineEditStatusChannelAnalysis.text(),
            '--output-channel', self.lineEditOutputChannelAnalysis.text(),
            '--processor-file', self.lineEditProcessorFileAnalysis.text(),
            '--processor-class', self.lineEditProcessorClassAnalysis.text(),
            '--report-period', str(self.spinBoxReportPeriodAnalysis.value()),
            '--server-queue-size', str(self.spinBoxServerQueueSizeAnalysis.value())
        ]

        # Add metadata config file if specified
        metadata_config = self.lineEditMetadataConfigAnalysis.text()
        if metadata_config:
            cmd.extend(['--metadata-config', metadata_config])

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                universal_newlines=True
            )
            self.processes['analysis_consumer'] = process

            # Start thread to read output
            worker = Worker(process)
            worker.output_signal.connect(self.textEditAnalysisConsumerOutput.appendPlainText)
            thread = threading.Thread(target=worker.run)
            thread.daemon = True
            thread.start()
            self.workers['analysis_consumer'] = (worker, thread)

            # Update button states and status label
            self.buttonRunAnalysisConsumer.setEnabled(False)
            self.buttonStopAnalysisConsumer.setEnabled(True)
            self.labelStatusAnalysisConsumer.setText(f'Process ID: {process.pid}')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to start Analysis Consumer: {str(e)}')

    def stop_analysis_consumer(self):
        if 'analysis_consumer' in self.processes:
            self.workers['analysis_consumer'][0].stop()
            self.processes['analysis_consumer'].wait()
            del self.processes['analysis_consumer']
            del self.workers['analysis_consumer']
            self.buttonRunAnalysisConsumer.setEnabled(True)
            self.buttonStopAnalysisConsumer.setEnabled(False)
            self.labelStatusAnalysisConsumer.setText('Process ID: Not running')
            self.textEditAnalysisConsumerOutput.appendPlainText('Analysis Consumer stopped.')

    def closeEvent(self, event):
        # Terminate all processes when the dialog is closed
        for key in list(self.processes.keys()):
            self.workers[key][0].stop()
            self.processes[key].wait()
            del self.processes[key]
            del self.workers[key]
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    dialog = PVASetupDialog()
    dialog.show()
    sys.exit(app.exec_())

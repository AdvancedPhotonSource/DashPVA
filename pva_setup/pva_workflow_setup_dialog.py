import sys
import subprocess
import threading
import os
import signal
import toml
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
        self.buttonRunSimServer.clicked.connect(self.run_sim_server)
        self.buttonStopSimServer.clicked.connect(self.stop_sim_server)

        # Associator Consumers Tab
        self.buttonBrowseMetadataAssociator.clicked.connect(self.browse_metadata_config_associator)
        self.buttonRunAssociatorConsumers.clicked.connect(self.run_associator_consumers)
        self.buttonStopAssociatorConsumers.clicked.connect(self.stop_associator_consumers)

        # Collector Tab
        self.buttonBrowseMetadataCollector.clicked.connect(self.browse_metadata_config_collector)
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
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Metadata Config File', '', 'TOML Files (*.toml)')
        if file_name:
            self.lineEditMetadataConfigSim.setText(file_name)

    # Browse functions for AssociatorConsumers
    def browse_processor_file_associator(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Associator Consumers Processor File', '', 'Python Files (*.py)')
        if file_name:
            self.lineEditProcessorFileAssociator.setText(file_name)

    def browse_metadata_config_associator(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Metadata Config File', '', 'TOML Files (*.toml)')
        if file_name:
            self.lineEditMetadataConfigAssociator.setText(file_name)

    # Browse functions for Collector
    def browse_processor_file_collector(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Collector Processor File', '', 'Python Files (*.py)')
        if file_name:
            self.lineEditProcessorFileCollector.setText(file_name)

    def browse_metadata_config_collector(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select ROI Config File', '', 'TOML Files (*.toml)')
        if file_name:
            self.lineEditMetadataConfigCollector.setText(file_name)

    # Browse functions for Analysis Consumer
    def browse_processor_file_analysis(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Analysis Consumer Processor File', '', 'Python Files (*.py)')
        if file_name:
            self.lineEditProcessorFileAnalysis.setText(file_name)

    def browse_metadata_config_analysis(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Metadata Config File', '', 'TOML Files (*.toml)')
        if file_name:
            self.lineEditMetadataConfigAnalysis.setText(file_name)

    def parse_toml(self, path) -> dict: 
        with open(path, 'r') as f:
            toml_data: dict = toml.load(f)
        return toml_data
    
    def parse_metadata_channels(self, metadata_config_path) -> str:
        pv_config = self.parse_toml(path=metadata_config_path)
        metadata_config : dict = pv_config.get("metadata", {})
        
        if metadata_config and metadata_config is not None:
            ca = metadata_config.get("ca", {})
            pva = metadata_config.get("pva", {})
            ca_pvs = ""
            pva_pvs = ""
            if ca:
                for value in list(ca.values()):
                    ca_pvs += f"ca://{value},"
            if pva:
                for value in list(pva.values()):
                    pva_pvs += f"pva://{value},"
            
        all_pvs = ca_pvs.strip(',') if not(pva_pvs) else ca_pvs + pva_pvs.strip(',')
        return all_pvs

    def parse_roi_channels(self, roi_config_path) -> str:
        pv_config = self.parse_toml(roi_config_path)
        roi_config: dict = pv_config.get("rois", {})
        #num_rois = len(roi_config)
        roi_pvs = ""

        if roi_config and roi_config is not None:
            for roi in roi_config.keys():
                roi_specific_pvs: dict = roi_config.get(roi, {})
                if roi_specific_pvs:
                    for pv in roi_specific_pvs.keys():
                        pv_channel = roi_specific_pvs.get(pv, "")
                        if pv_channel:
                            roi_pvs += f"ca://{pv_channel},"

        roi_pvs = roi_pvs.strip(',')
        return roi_pvs

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
            '-rp', str(self.spinBoxRp.value())
        ]

        metadata_output_pvs = self.lineEditMpv.text()
        if metadata_output_pvs:
            cmd.extend(['-mpv', metadata_output_pvs])
        
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

    # Run and Stop functions for Associator Consumers
    def run_associator_consumers(self):
        if 'associator_consumers' in self.processes:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Associator Consumers are already running.')
            return

        cmd = [
            'pvapy-hpc-consumer',
            '--input-channel', self.lineEditInputChannelAssociator.text(),
            '--control-channel', self.lineEditControlChannelAssociator.text(),
            '--status-channel', self.lineEditStatusChannelAssociator.text(),
            '--output-channel', self.lineEditOutputChannelAssociator.text(),
            '--processor-file', self.lineEditProcessorFileAssociator.text(),
            '--processor-class', self.lineEditProcessorClassAssociator.text(),
            '--report-period', str(self.spinBoxReportPeriodAssociator.value()),
            '--server-queue-size', str(self.spinBoxServerQueueSizeAssociator.value()),
            '--n-consumers', str(self.spinBoxNConsumersAssociator.value()),
            '--distributor-updates', str(self.spinBoxDistributorUpdatesAssociator.value()),
        ]

        # Add metadata config file if specified
        config_path = self.lineEditMetadataConfigAssociator.text()
        if config_path:
            metadata_config = self.parse_metadata_channels(config_path)
            if metadata_config:
                cmd.extend(['--metadata-channels', metadata_config])

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                universal_newlines=True
            )
            self.processes['associator_consumers'] = process

            # Start thread to read output
            worker = Worker(process)
            worker.output_signal.connect(self.textEditAssociatorConsumersOutput.appendPlainText)
            thread = threading.Thread(target=worker.run)
            thread.daemon = True
            thread.start()
            self.workers['associator_consumers'] = (worker, thread)

            # Update button states and status label
            self.buttonRunAssociatorConsumers.setEnabled(False)
            self.buttonStopAssociatorConsumers.setEnabled(True)
            self.labelStatusAssociatorConsumers.setText(f'Process ID: {process.pid}')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to start Associator Consumers: {str(e)}')

    def stop_associator_consumers(self):
        if 'associator_consumers' in self.processes:
            self.workers['associator_consumers'][0].stop()
            self.processes['associator_consumers'].wait()
            del self.processes['associator_consumers']
            del self.workers['associator_consumers']
            self.buttonRunAssociatorConsumers.setEnabled(True)
            self.buttonStopAssociatorConsumers.setEnabled(False)
            self.labelStatusAssociatorConsumers.setText('Process ID: Not running')
            self.textEditAssociatorConsumersOutput.appendPlainText('Associator Consumers stopped.')

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
        config_path = self.lineEditMetadataConfigCollector.text()
        if config_path:
            metadata_config = self.parse_roi_channels(config_path)
            if metadata_config:
                cmd.extend(['--metadata-channels', metadata_config])

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

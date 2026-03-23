import sys
import subprocess
import threading
import os
import signal
import toml
import json
from PyQt5 import QtWidgets, uic, QtCore, QtGui
from PyQt5.QtWidgets import (
    QFileDialog, QDialog, QTextEdit, QTreeWidgetItem, QHeaderView,
    QInputDialog, QAbstractItemView, QMessageBox, QFormLayout, QLineEdit, QDialogButtonBox,
    QButtonGroup
)
from PyQt5.QtCore import pyqtSignal, QObject, Qt
from datetime import datetime
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from utils import PVAReader
from utils.log_manager import LogMixin
from database.interface import DatabaseInterface
import settings as app_settings
from functools import partial


class Worker(QObject):
    """
    Worker class to manage subprocess output and communicate it back to the main thread.
    """
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
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        except Exception:
            pass


class ProfileEditDialog(QDialog):
    """Small dialog for editing a profile name and description together."""
    def __init__(self, name='', description='', parent=None):
        super().__init__(parent)
        self.setWindowTitle('Edit Profile')
        self.setMinimumWidth(360)
        layout = QFormLayout(self)
        self._name_edit = QLineEdit(name)
        self._desc_edit = QLineEdit(description)
        self._desc_edit.setPlaceholderText('Optional description')
        layout.addRow('Name:', self._name_edit)
        layout.addRow('Description:', self._desc_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    @property
    def name(self):
        return self._name_edit.text().strip()

    @property
    def description(self):
        return self._desc_edit.text().strip()


class JsonEditorDialog(QDialog):
    """Modal JSON editor with a Simple tree view and a Raw JSON text view."""

    def __init__(self, value=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Edit JSON Value')
        self.setMinimumSize(560, 500)
        self._value = value if value is not None else {}

        main_layout = QtWidgets.QVBoxLayout(self)

        # ── Tab widget ──────────────────────────────────────────────────
        self._tabs = QtWidgets.QTabWidget()

        # Simple tab
        simple_widget = QtWidgets.QWidget()
        simple_layout = QtWidgets.QVBoxLayout(simple_widget)
        simple_layout.setContentsMargins(4, 4, 4, 4)

        self._tree = QtWidgets.QTreeWidget()
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(['Key', 'Value'])
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self._tree.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._tree_context_menu)
        simple_layout.addWidget(self._tree)

        tree_btn_row = QtWidgets.QHBoxLayout()
        for label, slot in [
            ('Add Key',       self._add_key),
            ('Add Child Key', self._add_child_key),
            ('Delete',        self._delete_key),
        ]:
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(slot)
            tree_btn_row.addWidget(b)
        tree_btn_row.addStretch()
        simple_layout.addLayout(tree_btn_row)
        self._tabs.addTab(simple_widget, 'Simple')

        # Raw tab
        raw_widget = QtWidgets.QWidget()
        raw_layout = QtWidgets.QVBoxLayout(raw_widget)
        raw_layout.setContentsMargins(4, 4, 4, 4)

        self._editor = QtWidgets.QPlainTextEdit()
        self._editor.setFont(QtGui.QFont('Courier', 10))
        self._editor.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        raw_layout.addWidget(self._editor)

        fmt_row = QtWidgets.QHBoxLayout()
        fmt_btn = QtWidgets.QPushButton('Format')
        fmt_btn.clicked.connect(self._format)
        fmt_row.addWidget(fmt_btn)
        fmt_row.addStretch()
        raw_layout.addLayout(fmt_row)
        self._tabs.addTab(raw_widget, 'Raw JSON')

        self._tabs.currentChanged.connect(self._on_tab_changed)
        main_layout.addWidget(self._tabs)

        # Error label
        self._error_label = QtWidgets.QLabel()
        self._error_label.setStyleSheet('color: #E74C3C; font-size: 11px;')
        self._error_label.setVisible(False)
        main_layout.addWidget(self._error_label)

        # OK / Cancel
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        main_layout.addWidget(btns)

        # Seed both views
        self._populate_tree(self._value)
        self._editor.setPlainText(json.dumps(self._value, indent=2))

    # ── Tree helpers ────────────────────────────────────────────────────

    def _populate_tree(self, data, parent=None):
        if parent is None:
            self._tree.clear()
        for key, value in (data.items() if isinstance(data, dict) else []):
            item = QTreeWidgetItem(self._tree if parent is None else parent, [str(key), ''])
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            if isinstance(value, dict):
                self._populate_tree(value, item)
                item.setExpanded(True)
            else:
                if value is None:
                    display = 'null'
                elif isinstance(value, bool):
                    display = 'true' if value else 'false'
                elif isinstance(value, (list, dict)):
                    display = json.dumps(value)
                else:
                    display = str(value)
                item.setText(1, display)

    def _tree_to_dict(self, parent=None):
        result = {}
        if parent is None:
            items = [self._tree.topLevelItem(i) for i in range(self._tree.topLevelItemCount())]
        else:
            items = [parent.child(i) for i in range(parent.childCount())]
        for item in items:
            key = item.text(0)
            result[key] = self._tree_to_dict(item) if item.childCount() > 0 else self._coerce(item.text(1))
        return result

    @staticmethod
    def _coerce(text):
        if text == 'null':  return None
        if text == 'true':  return True
        if text == 'false': return False
        try: return int(text)
        except ValueError: pass
        try: return float(text)
        except ValueError: pass
        try:
            v = json.loads(text)
            if isinstance(v, (list, dict)):
                return v
        except (json.JSONDecodeError, ValueError):
            pass
        return text

    def _new_item(self, parent=None, key='key', value=''):
        item = QTreeWidgetItem(self._tree if parent is None else parent, [key, value])
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        return item

    # ── Tree actions ────────────────────────────────────────────────────

    def _add_key(self):
        selected = self._tree.currentItem()
        parent = selected.parent() if selected else None
        item = self._new_item(parent)
        self._tree.setCurrentItem(item)
        self._tree.editItem(item, 0)

    def _add_child_key(self):
        selected = self._tree.currentItem()
        item = self._new_item(selected)   # root if selected is None
        if selected:
            selected.setExpanded(True)
        self._tree.setCurrentItem(item)
        self._tree.editItem(item, 0)

    def _delete_key(self):
        selected = self._tree.currentItem()
        if selected is None:
            return
        parent = selected.parent()
        if parent is None:
            self._tree.takeTopLevelItem(self._tree.indexOfTopLevelItem(selected))
        else:
            parent.removeChild(selected)

    def _tree_context_menu(self, pos):
        item = self._tree.itemAt(pos)
        menu = QtWidgets.QMenu(self)
        menu.addAction('Add Key').triggered.connect(self._add_key)
        menu.addAction('Add Child Key').triggered.connect(self._add_child_key)
        if item:
            menu.addSeparator()
            menu.addAction('Delete').triggered.connect(self._delete_key)
        menu.exec_(self._tree.viewport().mapToGlobal(pos))

    # ── Tab sync ────────────────────────────────────────────────────────

    def _on_tab_changed(self, index):
        self._error_label.setVisible(False)
        if index == 0:   # → Simple: parse raw text
            try:
                data = json.loads(self._editor.toPlainText())
                self._populate_tree(data)
            except json.JSONDecodeError as e:
                self._error_label.setText(f'Cannot switch — invalid JSON: {e}')
                self._error_label.setVisible(True)
                self._tabs.blockSignals(True)
                self._tabs.setCurrentIndex(1)
                self._tabs.blockSignals(False)
        else:            # → Raw: serialize tree
            self._editor.setPlainText(json.dumps(self._tree_to_dict(), indent=2))

    # ── Raw tab ─────────────────────────────────────────────────────────

    def _format(self):
        try:
            self._editor.setPlainText(json.dumps(json.loads(self._editor.toPlainText()), indent=2))
            self._error_label.setVisible(False)
        except json.JSONDecodeError as e:
            self._error_label.setText(f'Invalid JSON: {e}')
            self._error_label.setVisible(True)

    # ── Accept ───────────────────────────────────────────────────────────

    def _on_accept(self):
        if self._tabs.currentIndex() == 0:
            self._value = self._tree_to_dict()
            self.accept()
        else:
            try:
                self._value = json.loads(self._editor.toPlainText())
                self._error_label.setVisible(False)
                self.accept()
            except json.JSONDecodeError as e:
                self._error_label.setText(f'Invalid JSON: {e}')
                self._error_label.setVisible(True)

    @property
    def value(self):
        return self._value


class Workflow(QDialog, LogMixin):
    """
    Dialog for setting up and managing the PVA workflow.
    """
    def __init__(self, parent=None):
        super(Workflow, self).__init__(parent)
        uic.loadUi(str(pathlib.Path(__file__).parent / 'workflow.ui'), self)
        try:
            self.set_log_manager(viewer_name="Workflow")
        except Exception:
            pass

        self.processes = {}
        self.workers = {}
        self._db = None
        self._db_available = False

        # Sim Server Tab
        self.buttonRunSimServer.clicked.connect(self.run_sim_server)
        self.buttonStopSimServer.clicked.connect(self.stop_sim_server)

        # Config Tab — source radio (explicit group ensures mutual exclusivity)
        self._config_source_group = QButtonGroup(self)
        self._config_source_group.addButton(self.radioLegacyToml)
        self._config_source_group.addButton(self.radioDatabase)
        self.radioLegacyToml.toggled.connect(self._on_config_source_changed)
        self.radioDatabase.toggled.connect(self._on_config_source_changed)

        # Config Tab — view mode toggle (Profile vs Settings)
        self._view_mode_group = QButtonGroup(self)
        self._view_mode_group.addButton(self.radioViewProfile)
        self._view_mode_group.addButton(self.radioViewSettings)
        self.radioViewProfile.toggled.connect(self._on_view_mode_changed)
        self.radioViewSettings.toggled.connect(self._on_view_mode_changed)

        # Config Tab — file-based
        self.buttonBrowseConfigUpload.clicked.connect(self.browse_config_upload)
        self.buttonImportToml.clicked.connect(self.import_toml_to_tree)

        # Config Tab — db-based
        self.buttonAddProfile.clicked.connect(self.add_profile)
        self.buttonDeleteProfile.clicked.connect(self.delete_profile)
        self.buttonDuplicateProfile.clicked.connect(self.duplicate_profile)
        self.comboBoxProfile.currentIndexChanged.connect(self._on_profile_selected)
        self.buttonRenameProfile.clicked.connect(self.edit_profile)
        self.checkBoxDefaultProfile.toggled.connect(self._on_default_toggled)
        self.checkBoxSelectedProfile.toggled.connect(self._on_selected_toggled)

        # Config Tab — always available
        self.buttonExportConfigToFile.clicked.connect(self.export_config_to_file)
        self.lineEditTreeSearch.textChanged.connect(self._filter_tree)

        # Config Tab — apply/save
        self.buttonApplySave.clicked.connect(self._on_apply_save)

        self.treeWidgetConfig.itemChanged.connect(self._on_tree_item_changed)
        self.treeWidgetConfig.itemDoubleClicked.connect(self._on_tree_item_double_clicked)
        self.treeWidgetConfig.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.treeWidgetConfig.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.treeWidgetConfig.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.treeWidgetConfig.setContextMenuPolicy(Qt.CustomContextMenu)
        self.treeWidgetConfig.customContextMenuRequested.connect(self._show_tree_context_menu)
        self.buttonCollapseAll.clicked.connect(self.treeWidgetConfig.collapseAll)
        self.buttonExpandAll.clicked.connect(self.treeWidgetConfig.expandAll)
        self.buttonAddTopLevel.clicked.connect(self._add_top_level_key)

        # Associator Consumers Tab
        self.buttonRunAssociatorConsumers.clicked.connect(self.run_associator_consumers)
        self.buttonStopAssociatorConsumers.clicked.connect(self.stop_associator_consumers)

        # Collector Tab
        self.buttonRunCollector.clicked.connect(self.run_collector)
        self.buttonStopCollector.clicked.connect(self.stop_collector)

        # Analysis Consumer Tab
        self.buttonRunAnalysisConsumer.clicked.connect(self.run_analysis_consumer)
        self.buttonStopAnalysisConsumer.clicked.connect(self.stop_analysis_consumer)

        # DB init / refresh buttons
        self.buttonInitDb.clicked.connect(self._init_db_and_recheck)
        self.buttonRefreshDb.clicked.connect(self._refresh_db)

        # Named config save / load buttons
        self.buttonSaveConfigMeta.clicked.connect(
            lambda: self._save_named_config_dialog(self._META_ASSOC_PATH, self._collect_meta_assoc_config)
        )
        self.buttonLoadConfigMeta.clicked.connect(
            lambda: self._load_named_config_dialog(self._META_ASSOC_PATH, self._apply_meta_assoc_config)
        )
        self.buttonSaveConfigCollector.clicked.connect(
            lambda: self._save_named_config_dialog(self._COLLECTOR_PATH, self._collect_collector_config)
        )
        self.buttonLoadConfigCollector.clicked.connect(
            lambda: self._load_named_config_dialog(self._COLLECTOR_PATH, self._apply_collector_config)
        )
        self.buttonSaveConfigAnalysis.clicked.connect(
            lambda: self._save_named_config_dialog(self._ANALYSIS_PATH, self._collect_analysis_config)
        )
        self.buttonLoadConfigAnalysis.clicked.connect(
            lambda: self._load_named_config_dialog(self._ANALYSIS_PATH, self._apply_analysis_config)
        )

        # Check DB availability, populate combo, then switch to DB mode if possible
        self._check_db_availability()
        self._on_config_source_changed()   # sets Legacy mode UI state
        self._refresh_profile_combo()      # populate combo, pre-select right profile
        if self._db_available:
            self.radioDatabase.setChecked(True)  # triggers _on_config_source_changed → load
        self._populate_processor_file_combos()
        self._load_meta_assoc_last()
        self._load_collector_last()
        self._load_analysis_last()

    # ------------------------------------------------------------------ #
    # DB availability check
    # ------------------------------------------------------------------ #

    def _check_db_availability(self):
        try:
            self._db = DatabaseInterface()
            self._db.get_all_profiles()  # confirm tables are queryable
            self._db_available = True
            self.labelDbStatus.setText('● Available')
            self.labelDbStatus.setStyleSheet('QLabel { color: #27AE60; font-size: 10px; margin-left: 4px; }')
            self.labelDbStatus.setToolTip('')
            self.radioDatabase.setEnabled(True)
            self.buttonInitDb.setVisible(False)
        except Exception as e:
            self._db_available = False
            self.labelDbStatus.setText('● Unavailable')
            self.labelDbStatus.setStyleSheet('QLabel { color: #E74C3C; font-size: 10px; margin-left: 4px; }')
            self.labelDbStatus.setToolTip(f'Error: {e}')
            self.radioDatabase.setEnabled(False)
            self.buttonInitDb.setVisible(True)
            # Fall back to legacy TOML if database radio was selected
            self.radioLegacyToml.setChecked(True)

    def _init_db_and_recheck(self):
        from database.db import init_database, create_tables
        errors = []
        try:
            init_database()
        except Exception as e:
            errors.append(f'init_database: {e}')
        try:
            create_tables()
        except Exception as e:
            errors.append(f'create_tables: {e}')
        self._check_db_availability()
        if not self._db_available:
            detail = '\n'.join(errors) if errors else self.labelDbStatus.toolTip()
            QMessageBox.warning(self, 'DB Initialization Failed', f'Could not initialize database.\n\n{detail}')

    def _refresh_db(self):
        """Re-check DB, reload the active view, and re-populate meta assoc fields."""
        self._check_db_availability()
        if self._db_available:
            self._refresh_profile_combo()
            if self.radioViewSettings.isChecked():
                self._load_settings_tree()
            self._populate_processor_file_combos()
            self._load_meta_assoc_last()
            self._load_collector_last()
            self._load_analysis_last()

    def _populate_processor_file_combos(self):
        """Populate processor file dropdowns from CONSUMERS > hpc in the DB."""
        if not self._db_available:
            return
        try:
            # Look up CONSUMERS by name — works whether it's root or under PATHS
            consumers_setting = self._db.get_setting_by_name('CONSUMERS')
            if consumers_setting is None:
                return
            consumers_base = self._db.get_setting_value(consumers_setting.id, 'BASE') or ''

            hpc_setting = next(
                (c for c in self._db.get_setting_children(consumers_setting.id) if c.name == 'hpc'),
                None
            )
            if hpc_setting is None:
                return
            hpc_base     = self._db.get_setting_value(hpc_setting.id, 'BASE')     or ''
            meta_dir     = self._db.get_setting_value(hpc_setting.id, 'meta')     or ''
            analysis_dir = self._db.get_setting_value(hpc_setting.id, 'analysis') or ''
        except Exception:
            return

        # Resolve relative to the project root so cwd doesn't matter
        project_root = pathlib.Path(__file__).parent.parent

        def list_py_files(subdir):
            try:
                d = project_root / consumers_base / hpc_base / subdir if subdir else project_root / consumers_base / hpc_base
                if not d.is_dir():
                    return []
                rel_base = pathlib.Path(consumers_base) / hpc_base / subdir
                return sorted(str(rel_base / f.name) for f in sorted(d.glob('*.py')))
            except Exception:
                return []

        meta_files     = list_py_files(meta_dir)
        analysis_files = list_py_files(analysis_dir)

        for combo, files in [
            (self.comboBoxProcessorFileAssociator, meta_files),
            (self.comboBoxProcessorFileCollector,  meta_files),
            (self.comboBoxProcessorFileAnalysis,   analysis_files),
        ]:
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(files)
            if current:
                combo.setCurrentText(current)
            combo.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Config source toggle
    # ------------------------------------------------------------------ #

    def _on_config_source_changed(self):
        legacy = self.radioLegacyToml.isChecked()

        # Legacy TOML controls
        self.lineEditConfigUploadPath.setEnabled(legacy)
        self.buttonBrowseConfigUpload.setEnabled(legacy)

        # Import TOML is available in both modes
        self.buttonImportToml.setEnabled(True)

        # Database controls (only enable if DB is also available)
        db_active = not legacy and self._db_available
        self.comboBoxProfile.setEnabled(db_active)
        self.checkBoxDefaultProfile.setEnabled(db_active)
        self.checkBoxSelectedProfile.setEnabled(db_active)
        self.radioViewSettings.setEnabled(db_active)
        self.buttonAddProfile.setEnabled(db_active)
        self.buttonRenameProfile.setEnabled(db_active)
        self.buttonDeleteProfile.setEnabled(db_active)
        self.buttonDuplicateProfile.setEnabled(db_active)

        # When first activating DB mode, auto-select the first profile as default+selected
        if db_active:
            self._auto_select_first_profile()
        if db_active and self.comboBoxProfile.currentIndex() >= 0:
            self.load_profile_to_tree()

    # ------------------------------------------------------------------ #
    # Config tab — profile combo
    # ------------------------------------------------------------------ #

    def _auto_select_first_profile(self):
        """When the DB is first activated, ensure a profile is marked default+selected.
        If exactly one profile exists, always ensure it is default+selected."""
        if not self._db_available:
            return
        try:
            profiles = self._db.get_all_profiles()
            if not profiles:
                return
            if len(profiles) == 1:
                first = profiles[0]
                # Always ensure the sole profile is default and selected
                needs_update = (
                    not getattr(first, 'is_default', False)
                    or not getattr(first, 'is_selected', False)
                )
                if needs_update:
                    self._db.set_default_profile(first.id)
                    self._db.set_selected_profile(first.id)
                    self._refresh_profile_combo()
                    idx = self.comboBoxProfile.findData(first.id)
                    if idx >= 0:
                        self.comboBoxProfile.setCurrentIndex(idx)
            elif not self._db.any_default_exists():
                first = profiles[0]
                self._db.set_default_profile(first.id)
                self._db.set_selected_profile(first.id)
                self._refresh_profile_combo()
                idx = self.comboBoxProfile.findData(first.id)
                if idx >= 0:
                    self.comboBoxProfile.setCurrentIndex(idx)
        except Exception:
            pass

    def _refresh_profile_combo(self):
        self.comboBoxProfile.blockSignals(True)
        self.comboBoxProfile.clear()
        if not self._db_available:
            self.comboBoxProfile.blockSignals(False)
            return
        try:
            profiles = self._db.get_all_profiles()
            if not profiles:
                self.comboBoxProfile.addItem("No Profiles Available")
                item = self.comboBoxProfile.model().item(0)
                if item:
                    item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                self.comboBoxProfile.setCurrentIndex(-1)
            else:
                target_idx = 0
                for i, p in enumerate(profiles):
                    self.comboBoxProfile.addItem(p.name, userData=p.id)
                    if getattr(p, 'is_selected', False):
                        target_idx = i
                    elif getattr(p, 'is_default', False) and not any(
                        getattr(profiles[j], 'is_selected', False) for j in range(i + 1)
                    ):
                        target_idx = i
                self.comboBoxProfile.setCurrentIndex(target_idx)
        except Exception:
            pass
        finally:
            self.comboBoxProfile.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Config tab — legacy TOML
    # ------------------------------------------------------------------ #

    def browse_config_upload(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, 'Select Config File', '', 'TOML Files (*.toml)'
        )
        if file_name:
            self.lineEditConfigUploadPath.setText(file_name)
            self.update_current_mode_label(file_name)
            self._load_toml_into_tree(file_name)

    def import_toml_to_tree(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, 'Import TOML Config', '', 'TOML Files (*.toml)'
        )
        if not file_name:
            return
        self.lineEditConfigUploadPath.setText(file_name)
        self.update_current_mode_label(file_name)

        # Always save to DB (if available) under the filename as profile name
        if self._db_available:
            profile_name = pathlib.Path(file_name).stem
            try:
                data = self.parse_toml(file_name)
                # Always create a new profile with a unique name
                candidate = profile_name
                n = 1
                while self._db.get_profile_by_name(candidate) is not None:
                    candidate = f'{profile_name}-({n})'
                    n += 1
                profile = self._db.create_profile(candidate)
                profile_name = candidate
                self._db.import_toml_to_profile(profile.id, data)
                # If this is the only profile, auto-mark it default+selected
                if profile is not None and len(self._db.get_all_profiles()) == 1:
                    self._db.set_default_profile(profile.id)
                    self._db.set_selected_profile(profile.id)
                self._refresh_profile_combo()
                idx = self.comboBoxProfile.findText(profile_name)
                if idx >= 0:
                    self.comboBoxProfile.setCurrentIndex(idx)
                # Switch to DB mode so the tree loads from the saved profile
                self.radioDatabase.setChecked(True)
                # Explicitly populate tree — setChecked is a no-op if already checked,
                # and setCurrentIndex won't fire if index didn't change (first import case)
                if self.comboBoxProfile.currentIndex() >= 0:
                    self.load_profile_to_tree()
                return
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to save TOML to database:\n{e}')
                # Fall through to load into tree without DB save

        # DB unavailable — just display in tree
        self._load_toml_into_tree(file_name)

    def _load_toml_into_tree(self, path: str):
        try:
            data = self.parse_toml(path)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to parse TOML:\n{e}')
            return
        self.treeWidgetConfig.clear()
        self._populate_tree_node(data, parent=None)
        # Update settings module so the rest of the app uses this TOML
        try:
            app_settings.set_locator(path)
            app_settings.reload()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Config tab — database profile
    # ------------------------------------------------------------------ #

    def _on_profile_selected(self, index):
        """Auto-load the selected profile into the tree when DB mode is active."""
        if not self.radioDatabase.isChecked() or not self._db_available:
            return
        if index < 0:
            return
        self.load_profile_to_tree()

    def load_profile_to_tree(self):
        idx = self.comboBoxProfile.currentIndex()
        if idx < 0:
            self.treeWidgetConfig.clear()
            return
        profile_id = self.comboBoxProfile.itemData(idx)
        try:
            profile = self._db.get_profile_by_id(profile_id)
            desc = getattr(profile, 'description', '') or ''
            self.lineEditProfileDescription.setText(desc)
            # Update checkboxes without triggering their toggle slots
            self.checkBoxDefaultProfile.blockSignals(True)
            self.checkBoxSelectedProfile.blockSignals(True)
            self.checkBoxDefaultProfile.setChecked(bool(getattr(profile, 'is_default', False)))
            self.checkBoxSelectedProfile.setChecked(bool(getattr(profile, 'is_selected', False)))
            self.checkBoxDefaultProfile.blockSignals(False)
            self.checkBoxSelectedProfile.blockSignals(False)
            # Show tooltips when this is the only profile
            is_sole = len(self._db.get_all_profiles()) == 1
            if is_sole:
                self.checkBoxDefaultProfile.setToolTip(
                    'Only profile — automatically set as default'
                )
                self.checkBoxSelectedProfile.setToolTip(
                    'Only profile — automatically set as selected'
                )
            else:
                self.checkBoxDefaultProfile.setToolTip('')
                self.checkBoxSelectedProfile.setToolTip('')
            data = self._db.export_profile_to_toml(profile_id)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load profile:\n{e}')
            return
        self.treeWidgetConfig.blockSignals(True)
        self.treeWidgetConfig.clear()
        self._populate_tree_node(data, parent=None)
        self.treeWidgetConfig.blockSignals(False)
        # Update settings module so the rest of the app uses this profile
        try:
            app_settings.set_locator(profile_id)
            app_settings.reload()
        except Exception:
            pass

    def _on_default_toggled(self, checked: bool):
        idx = self.comboBoxProfile.currentIndex()
        if idx < 0 or not self._db_available:
            return
        profile_id = self.comboBoxProfile.itemData(idx)
        try:
            if checked:
                self._db.set_default_profile(profile_id)
            else:
                self._db.unset_default_profile(profile_id)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to update default:\n{e}')

    def _on_selected_toggled(self, checked: bool):
        idx = self.comboBoxProfile.currentIndex()
        if idx < 0 or not self._db_available:
            return
        profile_id = self.comboBoxProfile.itemData(idx)
        try:
            if checked:
                self._db.set_selected_profile(profile_id)
            else:
                self._db.clear_selected_profiles()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to update selected:\n{e}')

    # ------------------------------------------------------------------ #
    # View mode toggle (Profile / Settings)
    # ------------------------------------------------------------------ #

    def _set_profile_widgets_visible(self, visible: bool):
        for w in (
            self.labelDatabaseHeader,
            self.labelProfile, self.comboBoxProfile,
            self.checkBoxDefaultProfile, self.checkBoxSelectedProfile,
            self.labelProfileDescription, self.lineEditProfileDescription,
            self.buttonAddProfile, self.buttonDeleteProfile,
            self.buttonRenameProfile, self.buttonDuplicateProfile,
            self.buttonImportToml, self.buttonExportConfigToFile,
        ):
            w.setVisible(visible)

    def _on_view_mode_changed(self):
        settings_mode = self.radioViewSettings.isChecked()
        self._set_profile_widgets_visible(not settings_mode)
        if settings_mode:
            self._load_settings_tree()
        elif self.radioDatabase.isChecked() and self._db_available:
            self._restore_two_column_tree()
            self.treeWidgetConfig.setEditTriggers(QAbstractItemView.DoubleClicked)
            self.load_profile_to_tree()
        else:
            self._restore_two_column_tree()
            self.treeWidgetConfig.setEditTriggers(QAbstractItemView.DoubleClicked)
            self.treeWidgetConfig.clear()

    def _restore_two_column_tree(self):
        self.treeWidgetConfig.setColumnCount(2)
        self.treeWidgetConfig.setHeaderLabels(['Setting', 'Value'])
        self.treeWidgetConfig.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.treeWidgetConfig.header().setSectionResizeMode(1, QHeaderView.Stretch)

    def _load_settings_tree(self):
        self.treeWidgetConfig.blockSignals(True)
        self.treeWidgetConfig.clear()
        self.treeWidgetConfig.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.treeWidgetConfig.setColumnCount(3)
        self.treeWidgetConfig.setHeaderLabels(['Setting', 'Value', 'Type'])
        self.treeWidgetConfig.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.treeWidgetConfig.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.treeWidgetConfig.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        if not self._db_available:
            self.treeWidgetConfig.blockSignals(False)
            return
        try:
            roots = self._db.get_root_settings()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load settings:\n{e}')
            self.treeWidgetConfig.blockSignals(False)
            return
        for root in roots:
            self._add_setting_node(root.id, root.name, root.type, parent=None)
        self.treeWidgetConfig.blockSignals(False)

    def _add_setting_node(self, setting_id: int, name: str, type_: str, parent):
        if parent is None:
            item = QTreeWidgetItem(self.treeWidgetConfig, [name, '', type_])
        else:
            item = QTreeWidgetItem(parent, [name, '', type_])
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)   # section header — not editable
        item.setData(0, Qt.UserRole, setting_id)           # store for child-add operations
        try:
            values = self._db.get_all_setting_values_with_type(setting_id)
            for k, v, vtype in values:
                if vtype == 'json' and not isinstance(v, str):
                    display_val = json.dumps(v)
                else:
                    display_val = str(v) if v is not None else ''
                val_item = QTreeWidgetItem(item, [k, display_val, vtype or ''])
                # Make value items editable (column 1 = Value); store setting_id for save
                val_item.setFlags(val_item.flags() | Qt.ItemIsEditable)
                val_item.setData(0, Qt.UserRole, setting_id)
        except Exception:
            pass
        try:
            children = self._db.get_setting_children(setting_id)
            for child in children:
                self._add_setting_node(child.id, child.name, child.type, parent=item)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Config tab — profile management
    # ------------------------------------------------------------------ #

    def edit_profile(self):
        idx = self.comboBoxProfile.currentIndex()
        if idx < 0:
            QMessageBox.warning(self, 'No Profile', 'No profile selected.')
            return
        profile_id = self.comboBoxProfile.itemData(idx)
        old_name = self.comboBoxProfile.itemText(idx)
        old_desc = self.lineEditProfileDescription.text()
        dlg = ProfileEditDialog(name=old_name, description=old_desc, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return
        new_name = dlg.name
        new_desc = dlg.description
        if not new_name:
            return
        try:
            if new_name != old_name:
                self._db.update_profile_name(profile_id, new_name)
                self.comboBoxProfile.setItemText(idx, new_name)
            if new_desc != old_desc:
                self._db.update_profile_description(profile_id, new_desc)
                self.lineEditProfileDescription.setText(new_desc)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to update profile:\n{e}')

    def add_profile(self):
        name, ok = QInputDialog.getText(self, 'Add Profile', 'Profile name:')
        if not ok or not name.strip():
            return
        name = name.strip()
        try:
            profile = self._db.create_profile(name)
            # Seed the new profile with the default sample config
            _sample = pathlib.Path(__file__).resolve().parents[1] / 'pv_configs' / 'sample_config.toml'
            if _sample.exists() and profile is not None:
                try:
                    defaults = self.parse_toml(str(_sample))
                    self._db.import_toml_to_profile(profile.id, defaults)
                except Exception:
                    pass  # don't block profile creation if seeding fails
            # If this is the only profile, auto-mark it default+selected before refreshing
            if profile is not None and len(self._db.get_all_profiles()) == 1:
                self._db.set_default_profile(profile.id)
                self._db.set_selected_profile(profile.id)
            self._refresh_profile_combo()
            idx = self.comboBoxProfile.findText(name)
            if idx >= 0:
                self.comboBoxProfile.setCurrentIndex(idx)
            # setCurrentIndex won't fire if index didn't change (first-profile case),
            # so explicitly load the tree here
            if self.comboBoxProfile.currentIndex() >= 0:
                self.load_profile_to_tree()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to create profile:\n{e}')

    def delete_profile(self):
        idx = self.comboBoxProfile.currentIndex()
        if idx < 0:
            QMessageBox.warning(self, 'No Profile', 'No profile selected.')
            return
        name = self.comboBoxProfile.currentText()
        profile_id = self.comboBoxProfile.itemData(idx)
        reply = QMessageBox.question(
            self, 'Confirm Delete',
            f'Delete profile "{name}"? This cannot be undone.',
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        try:
            self._db.delete_profile(profile_id)
            self.treeWidgetConfig.clear()
            self.lineEditProfileDescription.clear()
            self._refresh_profile_combo()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to delete profile:\n{e}')

    def duplicate_profile(self):
        idx = self.comboBoxProfile.currentIndex()
        if idx < 0:
            QMessageBox.warning(self, 'No Profile', 'No profile selected to duplicate.')
            return
        src_name = self.comboBoxProfile.currentText()
        src_id = self.comboBoxProfile.itemData(idx)
        new_name = f'{src_name}-(copy)'
        try:
            new_profile = self._db.create_profile(new_name)
            self._db.clone_profile_configs(src_id, new_profile.id)
            self._refresh_profile_combo()
            new_idx = self.comboBoxProfile.findText(new_name)
            if new_idx >= 0:
                self.comboBoxProfile.setCurrentIndex(new_idx)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to duplicate profile:\n{e}')

    # ------------------------------------------------------------------ #
    # Config tab — export to file
    # ------------------------------------------------------------------ #

    def export_config_to_file(self):
        data = self._extract_tree_to_dict()
        if not data:
            QMessageBox.warning(self, 'Empty Config', 'The config tree is empty. Load a config first.')
            return
        file_name, _ = QFileDialog.getSaveFileName(
            self, 'Export Config to File', '', 'TOML Files (*.toml)'
        )
        if not file_name:
            return
        try:
            with open(file_name, 'w') as f:
                toml.dump(data, f)
            QMessageBox.information(self, 'Exported', f'Config exported to:\n{file_name}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to export:\n{e}')

    # ------------------------------------------------------------------ #
    # Tree search
    # ------------------------------------------------------------------ #

    def _filter_tree(self, text: str):
        text = text.strip().lower()
        root = self.treeWidgetConfig.invisibleRootItem()
        for i in range(root.childCount()):
            self._filter_item(root.child(i), text)

    def _filter_item(self, item, text: str) -> bool:
        """Hide/show item based on whether it or any descendant matches text.
        Returns True if the item should be visible."""
        if not text:
            item.setHidden(False)
            for i in range(item.childCount()):
                self._filter_item(item.child(i), text)
            return True

        self_match = text in item.text(0).lower() or text in item.text(1).lower()

        child_match = False
        for i in range(item.childCount()):
            if self._filter_item(item.child(i), text):
                child_match = True

        visible = self_match or child_match
        item.setHidden(not visible)

        # If the group node itself matches, ensure all its children are shown
        if self_match and item.childCount() > 0:
            for i in range(item.childCount()):
                self._show_subtree(item.child(i))

        return visible

    def _show_subtree(self, item):
        item.setHidden(False)
        for i in range(item.childCount()):
            self._show_subtree(item.child(i))

    # ------------------------------------------------------------------ #
    # Tree helpers
    # ------------------------------------------------------------------ #

    def _populate_tree_node(self, data: dict, parent):
        for key, value in data.items():
            if isinstance(value, dict):
                if parent is None:
                    item = QTreeWidgetItem(self.treeWidgetConfig, [key, ''])
                else:
                    item = QTreeWidgetItem(parent, [key, ''])
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                self._populate_tree_node(value, item)
            else:
                if parent is None:
                    item = QTreeWidgetItem(self.treeWidgetConfig, [key, str(value)])
                else:
                    item = QTreeWidgetItem(parent, [key, str(value)])
                item.setFlags(item.flags() | Qt.ItemIsEditable)

    # ------------------------------------------------------------------ #
    # Tree editing — add / delete keys, right-click context menu
    # ------------------------------------------------------------------ #

    def _add_tree_key_dialog(self, title='Add Key', type_choices=None, default_key='', default_type=None):
        """Name + optional Type dialog (no value step).

        Used for profile-view add and for editing section metadata.
        Returns (key, None, type_or_None, accepted).
        """
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        layout = QFormLayout(dlg)
        key_edit = QLineEdit(default_key)
        layout.addRow('Name:', key_edit)
        type_combo = None
        val_edit = None
        if type_choices:
            type_combo = QtWidgets.QComboBox()
            for t in type_choices:
                type_combo.addItem(t)
            if default_type and default_type in type_choices:
                type_combo.setCurrentText(default_type)
            layout.addRow('Type:', type_combo)
        else:
            val_edit = QLineEdit()
            val_edit.setPlaceholderText('(optional)')
            layout.addRow('Value:', val_edit)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addRow(btns)
        if dlg.exec_() != QDialog.Accepted:
            return None, None, None, False
        key = key_edit.text().strip()
        if not key:
            return None, None, None, False
        chosen_type = type_combo.currentText() if type_combo else None
        val = val_edit.text().strip() or None if val_edit else None
        return key, val, chosen_type, True

    def _settings_two_step_add(self, title='Add Key', container_label='section'):
        """Two-step dialog for adding items in the settings view.

        container_label: label shown as the first type option ('root' for top-level,
                         'section' for child additions).

        Step 1: Name + Type  (<container_label> | string | int | float | json)
        Step 2 depends on type:
            container_label   → no value dialog, returns is_section=True
            json              → JSON editor opens
            string/int/float  → simple value input dialog

        Returns (name, value_or_None, type_str, accepted, is_section).
        """
        TYPE_CHOICES = [container_label, 'string', 'int', 'float', 'json']

        # --- Step 1: Name + Type ---
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        layout = QFormLayout(dlg)
        name_edit = QLineEdit()
        type_combo = QtWidgets.QComboBox()
        for t in TYPE_CHOICES:
            type_combo.addItem(t)
        layout.addRow('Name:', name_edit)
        layout.addRow('Type:', type_combo)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addRow(btns)
        if dlg.exec_() != QDialog.Accepted:
            return None, None, None, False, False
        name = name_edit.text().strip()
        if not name:
            return None, None, None, False, False
        chosen_type = type_combo.currentText()

        # --- Step 2: Value ---
        if chosen_type == container_label:
            return name, None, container_label, True, True

        if chosen_type == 'json':
            try:
                json_dlg = JsonEditorDialog(value={}, parent=self)
                json_dlg.raise_()
                json_dlg.activateWindow()
                if json_dlg.exec_() != QDialog.Accepted:
                    return None, None, None, False, False
                return name, json.dumps(json_dlg.value), 'json', True, False
            except Exception as e:
                QMessageBox.critical(self, 'JSON Editor Error', f'Failed to open JSON editor:\n{e}')
                return None, None, None, False, False

        # string / int / float — plain value input
        val_dlg = QDialog(self)
        val_dlg.setWindowTitle(f'Set Value ({chosen_type})')
        val_layout = QFormLayout(val_dlg)
        val_edit = QLineEdit()
        val_layout.addRow('Value:', val_edit)
        val_btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        val_btns.accepted.connect(val_dlg.accept)
        val_btns.rejected.connect(val_dlg.reject)
        val_layout.addRow(val_btns)
        if val_dlg.exec_() != QDialog.Accepted:
            return None, None, None, False, False
        return name, val_edit.text().strip(), chosen_type, True, False

    def _make_tree_item(self, parent, key, value):
        """Create a correctly-flagged QTreeWidgetItem (section or leaf)."""
        col1 = value if value is not None else ''
        if parent is None:
            item = QTreeWidgetItem(self.treeWidgetConfig, [key, col1])
        else:
            item = QTreeWidgetItem(parent, [key, col1])
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        return item

    def _add_top_level_key(self):
        if self.radioViewSettings.isChecked():
            self._add_settings_view_key()
            return
        key, value, _, ok = self._add_tree_key_dialog('Add Top-Level Key')
        if not ok:
            return
        item = self._make_tree_item(None, key, value)
        self.treeWidgetConfig.scrollToItem(item)
        self._save_tree_to_active_profile()

    def _add_settings_view_key(self):
        """Create a new root-level Settings record (and optional value) via two-step dialog."""
        if not self._db_available:
            return
        name, value, chosen_type, ok, is_section = self._settings_two_step_add('Add Top-Level Key', container_label='root')
        if not ok or not name:
            return
        section_type = 'root' if is_section else (chosen_type or 'root')
        try:
            new_setting = self._db.create_setting(name=name, type_=section_type)
            if new_setting is None:
                raise RuntimeError('create_setting returned None')
            if not is_section and value is not None:
                self._db.add_setting_value(new_setting.id, key=name, value=value, value_type=chosen_type)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to create setting:\n{e}')
            return
        self.treeWidgetConfig.blockSignals(True)
        self._add_setting_node(new_setting.id, new_setting.name, new_setting.type, parent=None)
        self.treeWidgetConfig.blockSignals(False)
        new_top = self.treeWidgetConfig.topLevelItem(
            self.treeWidgetConfig.topLevelItemCount() - 1
        )
        if new_top:
            self.treeWidgetConfig.scrollToItem(new_top)

    def _add_child_key(self, parent_item):
        in_settings = self.radioViewSettings.isChecked()
        if in_settings:
            name, value, chosen_type, ok, is_section = self._settings_two_step_add('Add Child Key')
            if not ok or not name:
                return
            parent_setting_id = parent_item.data(0, Qt.UserRole)
            if is_section:
                # Create a child Settings record
                if parent_setting_id is None:
                    QMessageBox.warning(self, 'Cannot Add Section', 'Select a section to nest under.')
                    return
                try:
                    child = self._db.create_child_setting(parent_id=parent_setting_id, name=name, type_='section')
                    if child is None:
                        raise RuntimeError('create_child_setting returned None')
                except Exception as e:
                    QMessageBox.critical(self, 'Error', f'Failed to create section:\n{e}')
                    return
                self.treeWidgetConfig.blockSignals(True)
                self._add_setting_node(child.id, child.name, child.type, parent=parent_item)
                self.treeWidgetConfig.blockSignals(False)
            else:
                # Create a SettingValue under the parent Settings record
                if parent_setting_id is not None and value is not None:
                    try:
                        self._db.add_setting_value(parent_setting_id, name, value, chosen_type or 'string')
                    except Exception as e:
                        QMessageBox.critical(self, 'Error', f'Failed to add setting value:\n{e}')
                        return
                self.treeWidgetConfig.blockSignals(True)
                new_item = self._make_tree_item(parent_item, name, value)
                new_item.setData(0, Qt.UserRole, parent_setting_id)
                new_item.setText(2, chosen_type or 'string')
                self.treeWidgetConfig.blockSignals(False)
            parent_item.setExpanded(True)
        else:
            key, value, _, ok = self._add_tree_key_dialog('Add Child Key')
            if not ok:
                return
            if parent_item.childCount() == 0 and parent_item.text(1):
                parent_item.setText(1, '')
                parent_item.setFlags(parent_item.flags() | Qt.ItemIsEditable)
            self._make_tree_item(parent_item, key, value)
            parent_item.setExpanded(True)
            self._save_tree_to_active_profile()

    def _add_section(self, parent_item):
        """Create a new child Settings record under parent_item's setting (settings view only)."""
        parent_setting_id = parent_item.data(0, Qt.UserRole)
        if parent_setting_id is None:
            QMessageBox.warning(self, 'Cannot Add Section', 'Select a section to nest under.')
            return
        try:
            existing_types = self._db._settings_mgr.get_distinct_types()
        except Exception:
            existing_types = []
        setting_types = sorted({'custom', 'path', 'root'} | set(existing_types))
        key, _, chosen_type, ok = self._add_tree_key_dialog('Add Section', type_choices=setting_types)
        if not ok or not key:
            return
        try:
            child_setting = self._db.create_child_setting(
                parent_id=parent_setting_id, name=key, type_=chosen_type or 'custom'
            )
            if child_setting is None:
                raise RuntimeError('create_child_setting returned None')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to create section:\n{e}')
            return
        self.treeWidgetConfig.blockSignals(True)
        self._add_setting_node(child_setting.id, child_setting.name, child_setting.type, parent=parent_item)
        self.treeWidgetConfig.blockSignals(False)
        parent_item.setExpanded(True)

    def _edit_section(self, item):
        """Rename/retype an existing Settings record (section header) in settings view."""
        setting_id = item.data(0, Qt.UserRole)
        if setting_id is None:
            return
        current_name = item.text(0)
        current_type = item.text(2)
        try:
            existing_types = self._db._settings_mgr.get_distinct_types()
        except Exception:
            existing_types = []
        setting_types = sorted({'custom', 'path', 'root'} | set(existing_types))
        # Reuse the key dialog: key=name, value unused, type=section type
        new_name, _, new_type, ok = self._add_tree_key_dialog(
            'Edit Section',
            type_choices=setting_types,
            default_key=current_name,
            default_type=current_type,
        )
        if not ok or not new_name:
            return
        try:
            self._db.update_setting(setting_id, new_name, new_type or current_type)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to update section:\n{e}')
            return
        self.treeWidgetConfig.blockSignals(True)
        item.setText(0, new_name)
        item.setText(2, new_type or current_type)
        self.treeWidgetConfig.blockSignals(False)

    def _delete_tree_item(self, item):
        def _summarize(node, indent=0):
            prefix = '  ' * indent
            val = node.text(1)
            line = f'{prefix}{node.text(0)}' + (f': {val}' if val else '')
            lines = [line]
            for i in range(node.childCount()):
                lines.extend(_summarize(node.child(i), indent + 1))
            return lines

        lines = _summarize(item)
        detail = '\n'.join(lines[:20])
        if len(lines) > 20:
            detail += f'\n  … ({len(lines) - 20} more)'
        reply = QMessageBox.question(
            self, 'Delete Entry',
            f'Delete this entry?\n\n{detail}',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        if self.radioViewSettings.isChecked():
            self._delete_settings_item_from_db(item)

        parent = item.parent()
        if parent is None:
            self.treeWidgetConfig.invisibleRootItem().removeChild(item)
        else:
            parent.removeChild(item)

        if not self.radioViewSettings.isChecked():
            self._save_tree_to_active_profile()

    def _delete_settings_item_from_db(self, item):
        """Delete a settings tree item from the DB immediately."""
        if not self._db_available:
            return
        setting_id = item.data(0, Qt.UserRole)
        if setting_id is None:
            return
        is_value_item = bool(item.flags() & Qt.ItemIsEditable)
        try:
            if is_value_item:
                self._db.remove_setting_value(setting_id, item.text(0))
            else:
                self._db.delete_setting(setting_id)
        except Exception as e:
            QMessageBox.critical(self, 'Delete Error', f'Failed to delete from database:\n{e}')

    def _duplicate_tree_item(self, item):
        """Deep-copy a tree item and insert it after the original with a unique key name."""
        def _clone(src, new_parent, key_override=None):
            key = key_override if key_override is not None else src.text(0)
            if new_parent is None:
                dst = QTreeWidgetItem(self.treeWidgetConfig, [key, src.text(1)])
            else:
                dst = QTreeWidgetItem(new_parent, [key, src.text(1)])
            dst.setFlags(src.flags())
            for i in range(src.childCount()):
                _clone(src.child(i), dst)
            return dst

        # Collect sibling keys to ensure uniqueness
        parent = item.parent()
        sibling_root = parent if parent is not None else self.treeWidgetConfig.invisibleRootItem()
        sibling_keys = {sibling_root.child(i).text(0) for i in range(sibling_root.childCount())}

        base = item.text(0)
        candidate = f'{base} - copy'
        if candidate in sibling_keys:
            n = 2
            while f'{base} - copy ({n})' in sibling_keys:
                n += 1
            candidate = f'{base} - copy ({n})'

        clone = _clone(item, parent, key_override=candidate)

        # Insert clone right after the original
        if parent is None:
            root = self.treeWidgetConfig.invisibleRootItem()
            idx = root.indexOfChild(item)
            root.removeChild(clone)
            root.insertChild(idx + 1, clone)
        else:
            idx = parent.indexOfChild(item)
            parent.removeChild(clone)
            parent.insertChild(idx + 1, clone)

        self.treeWidgetConfig.scrollToItem(clone)
        self._save_tree_to_active_profile()

    def _show_tree_context_menu(self, pos):
        item = self.treeWidgetConfig.itemAt(pos)
        in_settings = self.radioViewSettings.isChecked()
        menu = QtWidgets.QMenu(self)
        if item is None:
            act = menu.addAction('Add top-level key')
            act.triggered.connect(self._add_top_level_key)
        else:
            is_section = in_settings and not (item.flags() & Qt.ItemIsEditable)
            if in_settings and item.text(2) == 'json':
                act_json = menu.addAction('Edit JSON…')
                act_json.triggered.connect(lambda checked=False, i=item: self._open_json_editor(i))
                menu.addSeparator()
            if is_section:
                act_edit_sec = menu.addAction('Edit section…')
                act_edit_sec.triggered.connect(lambda checked=False, i=item: self._edit_section(i))
                menu.addSeparator()
            act_child = menu.addAction('Add child key')
            act_child.triggered.connect(lambda checked=False, i=item: self._add_child_key(i))
            act_dup = menu.addAction('Duplicate')
            act_dup.triggered.connect(lambda checked=False, i=item: self._duplicate_tree_item(i))
            menu.addSeparator()
            act_del = menu.addAction('Delete')
            act_del.triggered.connect(lambda checked=False, i=item: self._delete_tree_item(i))
        menu.exec_(self.treeWidgetConfig.viewport().mapToGlobal(pos))

    # ------------------------------------------------------------------ #
    # Lock toggle
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Apply / Save
    # ------------------------------------------------------------------ #

    def _on_apply_save(self):
        reply = QMessageBox.question(
            self,
            'Save Changes',
            'Are you sure you want to save these changes?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        if self.radioViewSettings.isChecked():
            self._save_settings_from_tree()
        else:
            self._save_tree_to_active_profile()

    def _save_settings_from_tree(self):
        """Persist all edited setting values from the Settings tree back to the DB."""
        if not self._db_available:
            return
        errors = []
        self._save_setting_items(self.treeWidgetConfig.invisibleRootItem(), errors)
        if errors:
            QMessageBox.warning(
                self, 'Save Warnings',
                'Some values could not be saved:\n' + '\n'.join(errors[:10]),
            )

    def _save_setting_items(self, parent_item, errors):
        for i in range(parent_item.childCount()):
            item = parent_item.child(i)
            setting_id = item.data(0, Qt.UserRole)
            if setting_id is not None:
                # Value item: key = column 0, value = column 1
                key = item.text(0)
                value = item.text(1)
                try:
                    self._db.update_setting_value(setting_id, key, value)
                except Exception as e:
                    errors.append(f'{key}: {e}')
            self._save_setting_items(item, errors)

    def _on_tree_item_double_clicked(self, item, column):
        """Route double-clicks in settings view: JSON items open the editor; others use inline edit."""
        if not self.radioViewSettings.isChecked():
            return
        if item.text(2) == 'json':
            self._open_json_editor(item)
        elif item.flags() & Qt.ItemIsEditable:
            self.treeWidgetConfig.editItem(item, 1)

    def _open_json_editor(self, item):
        """Open the JSON editor dialog for a json-typed settings value item."""
        raw = item.text(1)
        try:
            current_value = json.loads(raw) if raw.strip() else {}
        except (json.JSONDecodeError, ValueError):
            current_value = {}
        try:
            dlg = JsonEditorDialog(value=current_value, parent=self)
            dlg.raise_()
            dlg.activateWindow()
        except Exception as e:
            QMessageBox.critical(self, 'JSON Editor Error', f'Failed to open JSON editor:\n{e}')
            return
        if dlg.exec_() != QDialog.Accepted:
            return
        new_value = dlg.value
        serialized = json.dumps(new_value)
        self.treeWidgetConfig.blockSignals(True)
        item.setText(1, serialized)
        self.treeWidgetConfig.blockSignals(False)
        self._save_settings_view_item(item, column=1)

    def _on_tree_item_changed(self, item, column):
        if self.radioViewSettings.isChecked():
            self._save_settings_view_item(item, column)
        else:
            self._save_tree_to_active_profile()

    def _save_settings_view_item(self, item, column):
        """Save an edited value in the Settings view back to the DB."""
        if column != 1:
            return
        if not self._db_available:
            return
        setting_id = item.data(0, Qt.UserRole)
        # New items added via dialog have no UserRole — resolve from parent section header
        if setting_id is None:
            parent = item.parent()
            if parent is not None:
                setting_id = parent.data(0, Qt.UserRole)
        if setting_id is None:
            return
        key = item.text(0)
        new_value = self._coerce_value(item.text(1))
        try:
            updated = self._db.update_setting_value(setting_id, key, new_value)
            if not updated:
                self._db.add_setting_value(setting_id, key, new_value)
                item.setData(0, Qt.UserRole, setting_id)
        except Exception as e:
            QMessageBox.critical(self, 'Save Error', f'Failed to save setting:\n{e}')

    def _save_tree_to_active_profile(self):
        """Persist current tree contents to the active DB profile. No-op in TOML mode or Settings view."""
        if not (self.radioDatabase.isChecked() and self._db_available):
            return
        if self.radioViewSettings.isChecked():
            return
        idx = self.comboBoxProfile.currentIndex()
        if idx < 0:
            return
        profile_id = self.comboBoxProfile.itemData(idx)
        try:
            data = self._extract_tree_to_dict()
            self._db.clear_profile_configs(profile_id)
            self._db.import_toml_to_profile(profile_id, data)
        except Exception as e:
            QMessageBox.critical(self, 'Save Error', f'Failed to save to profile:\n{e}')

    def _extract_tree_to_dict(self, parent=None) -> dict:
        result = {}
        if parent is None:
            root = self.treeWidgetConfig.invisibleRootItem()
        else:
            root = parent
        for i in range(root.childCount()):
            child = root.child(i)
            key = child.text(0)
            if child.childCount() > 0:
                result[key] = self._extract_tree_to_dict(child)
            else:
                result[key] = self._coerce_value(child.text(1))
        return result

    @staticmethod
    def _coerce_value(text: str):
        lower = text.lower()
        if lower == 'true':
            return True
        if lower == 'false':
            return False
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            pass
        return text

    # ------------------------------------------------------------------ #
    # Config parsing helpers
    # ------------------------------------------------------------------ #

    def _build_metadata_channels(self) -> str:
        """Build --metadata-channels string from app_settings."""
        ca_pvs = ''.join(f'ca://{v},' for v in app_settings.METADATA_CA.values() if v)
        pva_pvs = ''.join(f'pva://{v},' for v in app_settings.METADATA_PVA.values() if v)
        for pvs_dict in app_settings.HKL.values():
            if isinstance(pvs_dict, dict):
                for pv_channel in pvs_dict.values():
                    if pv_channel:
                        ca_pvs += f'ca://{pv_channel},'
        all_pvs = ca_pvs.strip(',') if not pva_pvs else ca_pvs + pva_pvs.strip(',')
        return all_pvs

    def _build_roi_channels(self) -> str:
        """Build --metadata-channels string from app_settings ROI section."""
        roi_pvs = ''
        for roi_specific_pvs in app_settings.ROI.values():
            if isinstance(roi_specific_pvs, dict):
                for pv_channel in roi_specific_pvs.values():
                    if pv_channel:
                        roi_pvs += f'ca://{pv_channel},'
        return roi_pvs.strip(',')

    def parse_toml(self, path) -> dict:
        with open(path, 'r') as f:
            return toml.load(f)

    def update_current_mode_label(self, path: str) -> None:
        text = '(none)'
        try:
            toml_data = self.parse_toml(path)
            mode = toml_data.get('CACHE_OPTIONS', {}).get('CACHING_MODE', '')
            text = mode if mode else '(none)'
        except Exception:
            text = '(error)'
        try:
            self.labelCurrentModeValue.setText(text)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Sim Server
    # ------------------------------------------------------------------ #

    def run_sim_server(self):
        if 'sim_server' in self.processes:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Sim Server is already running.')
            return

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
            worker = Worker(process)
            worker.output_signal.connect(self.textEditSimServerOutput.appendPlainText)
            thread = threading.Thread(target=worker.run)
            thread.daemon = True
            thread.start()
            self.workers['sim_server'] = (worker, thread)
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

    # ------------------------------------------------------------------ #
    # Named config save / load (shared helpers)
    # ------------------------------------------------------------------ #

    def _save_named_config_dialog(self, path, collect_fn):
        """Prompt for a name and save the collected config dict under that key."""
        if not self._db_available:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, 'Save Configuration', 'Configuration name:')
        if not ok or not name.strip():
            return
        name = name.strip()
        setting = self._db.get_setting_by_path(path)
        if setting is None:
            QMessageBox.warning(self, 'Error', f'Settings path not found:\n{" > ".join(path)}')
            return
        config = collect_fn()
        if not self._db.update_setting_value(setting.id, name, config, value_type='json'):
            self._db.add_setting_value(setting.id, name, config, value_type='json')
        QMessageBox.information(self, 'Saved', f'Configuration "{name}" saved.')

    def _load_named_config_dialog(self, path, apply_fn):
        """Show a list of saved configs and apply the chosen one."""
        if not self._db_available:
            return
        setting = self._db.get_setting_by_path(path)
        if setting is None:
            QMessageBox.warning(self, 'Error', f'Settings path not found:\n{" > ".join(path)}')
            return
        all_values = self._db.get_all_setting_values(setting.id)
        configs = {k: v for k, v in all_values.items() if k != 'last' and isinstance(v, dict)}
        if not configs:
            QMessageBox.information(self, 'No Configs', 'No saved configurations found.')
            return
        name, ok = QtWidgets.QInputDialog.getItem(
            self, 'Load Configuration', 'Select configuration:', sorted(configs.keys()), 0, False
        )
        if not ok or not name:
            return
        apply_fn(configs[name])

    # ------------------------------------------------------------------ #
    # Associator Consumers
    # ------------------------------------------------------------------ #

    _META_ASSOC_PATH  = ['APP_DATA', 'workflow', 'meta']
    _COLLECTOR_PATH   = ['APP_DATA', 'workflow', 'collector']
    _ANALYSIS_PATH    = ['APP_DATA', 'workflow', 'analysis']

    def _load_meta_assoc_last(self):
        """Populate associator UI fields from APP_DATA > workflow > meta > last in the DB."""
        if not self._db_available:
            return
        try:
            setting = self._db.get_setting_by_path(self._META_ASSOC_PATH)
            if setting is None:
                return
            last = self._db.get_setting_value(setting.id, 'last')
            if not isinstance(last, dict):
                return
            self.lineEditInputChannelAssociator.setText(last.get('input_channel', ''))
            self.lineEditControlChannelAssociator.setText(last.get('control_channel', ''))
            self.lineEditStatusChannelAssociator.setText(last.get('status_channel', ''))
            self.lineEditOutputChannelAssociator.setText(last.get('output_channel', ''))
            self.comboBoxProcessorFileAssociator.setCurrentText(last.get('processor_file', ''))
            self.lineEditProcessorClassAssociator.setText(last.get('processor_class', ''))
            if 'report_period' in last:
                self.spinBoxReportPeriodAssociator.setValue(int(last['report_period']))
            if 'server_queue_size' in last:
                self.spinBoxServerQueueSizeAssociator.setValue(int(last['server_queue_size']))
            if 'n_consumers' in last:
                self.spinBoxNConsumersAssociator.setValue(int(last['n_consumers']))
            if 'distributor_updates' in last:
                self.spinBoxDistributorUpdatesAssociator.setValue(int(last['distributor_updates']))
        except Exception:
            pass

    def _save_meta_assoc_last(self):
        """Save current associator UI fields to APP_DATA > workflow > meta > last in the DB."""
        if not self._db_available:
            return
        try:
            setting = self._db.get_setting_by_path(self._META_ASSOC_PATH)
            if setting is None:
                return
            config = {
                'input_channel':       self.lineEditInputChannelAssociator.text(),
                'control_channel':     self.lineEditControlChannelAssociator.text(),
                'status_channel':      self.lineEditStatusChannelAssociator.text(),
                'output_channel':      self.lineEditOutputChannelAssociator.text(),
                'processor_file':      self.comboBoxProcessorFileAssociator.currentText(),
                'processor_class':     self.lineEditProcessorClassAssociator.text(),
                'report_period':       self.spinBoxReportPeriodAssociator.value(),
                'server_queue_size':   self.spinBoxServerQueueSizeAssociator.value(),
                'n_consumers':         self.spinBoxNConsumersAssociator.value(),
                'distributor_updates': self.spinBoxDistributorUpdatesAssociator.value(),
            }
            if not self._db.update_setting_value(setting.id, 'last', config, value_type='json'):
                self._db.add_setting_value(setting.id, 'last', config, value_type='json')
        except Exception:
            pass

    def _collect_meta_assoc_config(self):
        return {
            'input_channel':       self.lineEditInputChannelAssociator.text(),
            'control_channel':     self.lineEditControlChannelAssociator.text(),
            'status_channel':      self.lineEditStatusChannelAssociator.text(),
            'output_channel':      self.lineEditOutputChannelAssociator.text(),
            'processor_file':      self.comboBoxProcessorFileAssociator.currentText(),
            'processor_class':     self.lineEditProcessorClassAssociator.text(),
            'report_period':       self.spinBoxReportPeriodAssociator.value(),
            'server_queue_size':   self.spinBoxServerQueueSizeAssociator.value(),
            'n_consumers':         self.spinBoxNConsumersAssociator.value(),
            'distributor_updates': self.spinBoxDistributorUpdatesAssociator.value(),
        }

    def _apply_meta_assoc_config(self, cfg):
        self.lineEditInputChannelAssociator.setText(cfg.get('input_channel', ''))
        self.lineEditControlChannelAssociator.setText(cfg.get('control_channel', ''))
        self.lineEditStatusChannelAssociator.setText(cfg.get('status_channel', ''))
        self.lineEditOutputChannelAssociator.setText(cfg.get('output_channel', ''))
        self.comboBoxProcessorFileAssociator.setCurrentText(cfg.get('processor_file', ''))
        self.lineEditProcessorClassAssociator.setText(cfg.get('processor_class', ''))
        if 'report_period' in cfg:
            self.spinBoxReportPeriodAssociator.setValue(int(cfg['report_period']))
        if 'server_queue_size' in cfg:
            self.spinBoxServerQueueSizeAssociator.setValue(int(cfg['server_queue_size']))
        if 'n_consumers' in cfg:
            self.spinBoxNConsumersAssociator.setValue(int(cfg['n_consumers']))
        if 'distributor_updates' in cfg:
            self.spinBoxDistributorUpdatesAssociator.setValue(int(cfg['distributor_updates']))

    # ------------------------------------------------------------------ #
    # Collector last-used save / load
    # ------------------------------------------------------------------ #

    def _load_collector_last(self):
        """Populate collector UI fields from APP_DATA > workflow > collector > last in the DB."""
        if not self._db_available:
            return
        try:
            setting = self._db.get_setting_by_path(self._COLLECTOR_PATH)
            if setting is None:
                return
            last = self._db.get_setting_value(setting.id, 'last')
            if not isinstance(last, dict):
                return
            self.lineEditInputChannelCollector.setText(last.get('input_channel', ''))
            self.lineEditControlChannelCollector.setText(last.get('control_channel', ''))
            self.lineEditStatusChannelCollector.setText(last.get('status_channel', ''))
            self.lineEditOutputChannelCollector.setText(last.get('output_channel', ''))
            self.comboBoxProcessorFileCollector.setCurrentText(last.get('processor_file', ''))
            self.lineEditProcessorClassCollector.setText(last.get('processor_class', ''))
            self.lineEditProducerIdList.setText(last.get('producer_id_list', ''))
            if 'collector_id' in last:
                self.spinBoxCollectorId.setValue(int(last['collector_id']))
            if 'report_period' in last:
                self.spinBoxReportPeriodCollector.setValue(int(last['report_period']))
            if 'server_queue_size' in last:
                self.spinBoxServerQueueSizeCollector.setValue(int(last['server_queue_size']))
            if 'collector_cache_size' in last:
                self.spinBoxCollectorCacheSize.setValue(int(last['collector_cache_size']))
        except Exception:
            pass

    def _save_collector_last(self):
        """Save current collector UI fields to APP_DATA > workflow > collector > last in the DB."""
        if not self._db_available:
            return
        try:
            setting = self._db.get_setting_by_path(self._COLLECTOR_PATH)
            if setting is None:
                return
            config = {
                'collector_id':       self.spinBoxCollectorId.value(),
                'producer_id_list':   self.lineEditProducerIdList.text(),
                'input_channel':      self.lineEditInputChannelCollector.text(),
                'control_channel':    self.lineEditControlChannelCollector.text(),
                'status_channel':     self.lineEditStatusChannelCollector.text(),
                'output_channel':     self.lineEditOutputChannelCollector.text(),
                'processor_file':     self.comboBoxProcessorFileCollector.currentText(),
                'processor_class':    self.lineEditProcessorClassCollector.text(),
                'report_period':      self.spinBoxReportPeriodCollector.value(),
                'server_queue_size':  self.spinBoxServerQueueSizeCollector.value(),
                'collector_cache_size': self.spinBoxCollectorCacheSize.value(),
            }
            if not self._db.update_setting_value(setting.id, 'last', config, value_type='json'):
                self._db.add_setting_value(setting.id, 'last', config, value_type='json')
        except Exception:
            pass

    def _collect_collector_config(self):
        return {
            'collector_id':         self.spinBoxCollectorId.value(),
            'producer_id_list':     self.lineEditProducerIdList.text(),
            'input_channel':        self.lineEditInputChannelCollector.text(),
            'control_channel':      self.lineEditControlChannelCollector.text(),
            'status_channel':       self.lineEditStatusChannelCollector.text(),
            'output_channel':       self.lineEditOutputChannelCollector.text(),
            'processor_file':       self.comboBoxProcessorFileCollector.currentText(),
            'processor_class':      self.lineEditProcessorClassCollector.text(),
            'report_period':        self.spinBoxReportPeriodCollector.value(),
            'server_queue_size':    self.spinBoxServerQueueSizeCollector.value(),
            'collector_cache_size': self.spinBoxCollectorCacheSize.value(),
        }

    def _apply_collector_config(self, cfg):
        self.lineEditInputChannelCollector.setText(cfg.get('input_channel', ''))
        self.lineEditControlChannelCollector.setText(cfg.get('control_channel', ''))
        self.lineEditStatusChannelCollector.setText(cfg.get('status_channel', ''))
        self.lineEditOutputChannelCollector.setText(cfg.get('output_channel', ''))
        self.comboBoxProcessorFileCollector.setCurrentText(cfg.get('processor_file', ''))
        self.lineEditProcessorClassCollector.setText(cfg.get('processor_class', ''))
        self.lineEditProducerIdList.setText(cfg.get('producer_id_list', ''))
        if 'collector_id' in cfg:
            self.spinBoxCollectorId.setValue(int(cfg['collector_id']))
        if 'report_period' in cfg:
            self.spinBoxReportPeriodCollector.setValue(int(cfg['report_period']))
        if 'server_queue_size' in cfg:
            self.spinBoxServerQueueSizeCollector.setValue(int(cfg['server_queue_size']))
        if 'collector_cache_size' in cfg:
            self.spinBoxCollectorCacheSize.setValue(int(cfg['collector_cache_size']))

    # ------------------------------------------------------------------ #
    # Analysis Consumer last-used save / load
    # ------------------------------------------------------------------ #

    def _load_analysis_last(self):
        """Populate analysis consumer UI fields from APP_DATA > workflow > analysis > last in the DB."""
        if not self._db_available:
            return
        try:
            setting = self._db.get_setting_by_path(self._ANALYSIS_PATH)
            if setting is None:
                return
            last = self._db.get_setting_value(setting.id, 'last')
            if not isinstance(last, dict):
                return
            self.lineEditInputChannelAnalysis.setText(last.get('input_channel', ''))
            self.lineEditControlChannelAnalysis.setText(last.get('control_channel', ''))
            self.lineEditStatusChannelAnalysis.setText(last.get('status_channel', ''))
            self.lineEditOutputChannelAnalysis.setText(last.get('output_channel', ''))
            self.comboBoxProcessorFileAnalysis.setCurrentText(last.get('processor_file', ''))
            self.lineEditProcessorClassAnalysis.setText(last.get('processor_class', ''))
            if 'report_period' in last:
                self.spinBoxReportPeriodAnalysis.setValue(int(last['report_period']))
            if 'server_queue_size' in last:
                self.spinBoxServerQueueSizeAnalysis.setValue(int(last['server_queue_size']))
            if 'n_consumers' in last:
                self.spinBoxNConsumersAnalysis.setValue(int(last['n_consumers']))
            if 'distributor_updates' in last:
                self.spinBoxDistributorUpdatesAnalysis.setValue(int(last['distributor_updates']))
        except Exception:
            pass

    def _save_analysis_last(self):
        """Save current analysis consumer UI fields to APP_DATA > workflow > analysis > last in the DB."""
        if not self._db_available:
            return
        try:
            setting = self._db.get_setting_by_path(self._ANALYSIS_PATH)
            if setting is None:
                return
            config = {
                'input_channel':      self.lineEditInputChannelAnalysis.text(),
                'control_channel':    self.lineEditControlChannelAnalysis.text(),
                'status_channel':     self.lineEditStatusChannelAnalysis.text(),
                'output_channel':     self.lineEditOutputChannelAnalysis.text(),
                'processor_file':     self.comboBoxProcessorFileAnalysis.currentText(),
                'processor_class':    self.lineEditProcessorClassAnalysis.text(),
                'report_period':      self.spinBoxReportPeriodAnalysis.value(),
                'server_queue_size':  self.spinBoxServerQueueSizeAnalysis.value(),
                'n_consumers':        self.spinBoxNConsumersAnalysis.value(),
                'distributor_updates': self.spinBoxDistributorUpdatesAnalysis.value(),
            }
            if not self._db.update_setting_value(setting.id, 'last', config, value_type='json'):
                self._db.add_setting_value(setting.id, 'last', config, value_type='json')
        except Exception:
            pass

    def _collect_analysis_config(self):
        return {
            'input_channel':       self.lineEditInputChannelAnalysis.text(),
            'control_channel':     self.lineEditControlChannelAnalysis.text(),
            'status_channel':      self.lineEditStatusChannelAnalysis.text(),
            'output_channel':      self.lineEditOutputChannelAnalysis.text(),
            'processor_file':      self.comboBoxProcessorFileAnalysis.currentText(),
            'processor_class':     self.lineEditProcessorClassAnalysis.text(),
            'report_period':       self.spinBoxReportPeriodAnalysis.value(),
            'server_queue_size':   self.spinBoxServerQueueSizeAnalysis.value(),
            'n_consumers':         self.spinBoxNConsumersAnalysis.value(),
            'distributor_updates': self.spinBoxDistributorUpdatesAnalysis.value(),
        }

    def _apply_analysis_config(self, cfg):
        self.lineEditInputChannelAnalysis.setText(cfg.get('input_channel', ''))
        self.lineEditControlChannelAnalysis.setText(cfg.get('control_channel', ''))
        self.lineEditStatusChannelAnalysis.setText(cfg.get('status_channel', ''))
        self.lineEditOutputChannelAnalysis.setText(cfg.get('output_channel', ''))
        self.comboBoxProcessorFileAnalysis.setCurrentText(cfg.get('processor_file', ''))
        self.lineEditProcessorClassAnalysis.setText(cfg.get('processor_class', ''))
        if 'report_period' in cfg:
            self.spinBoxReportPeriodAnalysis.setValue(int(cfg['report_period']))
        if 'server_queue_size' in cfg:
            self.spinBoxServerQueueSizeAnalysis.setValue(int(cfg['server_queue_size']))
        if 'n_consumers' in cfg:
            self.spinBoxNConsumersAnalysis.setValue(int(cfg['n_consumers']))
        if 'distributor_updates' in cfg:
            self.spinBoxDistributorUpdatesAnalysis.setValue(int(cfg['distributor_updates']))

    def run_associator_consumers(self):
        if 'associator_consumers' in self.processes:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Associator Consumers are already running.')
            return
        self._save_meta_assoc_last()

        cmd = [
            'pvapy-hpc-consumer',
            '--input-channel', self.lineEditInputChannelAssociator.text(),
            '--control-channel', self.lineEditControlChannelAssociator.text(),
            '--status-channel', self.lineEditStatusChannelAssociator.text(),
            '--output-channel', self.lineEditOutputChannelAssociator.text(),
            '--processor-file', self.comboBoxProcessorFileAssociator.currentText(),
            '--processor-class', self.lineEditProcessorClassAssociator.text(),
            '--report-period', str(self.spinBoxReportPeriodAssociator.value()),
            '--server-queue-size', str(self.spinBoxServerQueueSizeAssociator.value()),
            '--n-consumers', str(self.spinBoxNConsumersAssociator.value()),
            '--distributor-updates', str(self.spinBoxDistributorUpdatesAssociator.value()),
            '-dc'
        ]

        metadata_pvs = self._build_metadata_channels()
        if metadata_pvs:
            cmd.extend(['--metadata-channels', metadata_pvs])

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                universal_newlines=True
            )
            self.processes['associator_consumers'] = process
            worker = Worker(process)
            output_format = partial(self._format_and_append_output, target_widget=self.textEditAssociatorConsumersOutput)
            worker.output_signal.connect(output_format)
            thread = threading.Thread(target=worker.run)
            thread.daemon = True
            thread.start()
            self.workers['associator_consumers'] = (worker, thread)
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

    def _format_and_append_output(self, text: str, target_widget: QTextEdit):
        timestamp = datetime.now().strftime('%H:%M:%S')
        safe_text = text.replace('<', '&lt;').replace('>', '&gt;')
        color = "#000000"
        if "ERROR" in text.upper():
            color = "#FF5733"
        elif "WARNING" in text.upper():
            color = "#FFC300"
        elif "SUCCESS" in text.upper() or "done" in text.lower():
            color = "#33FF57"
        formatted_line = f"<font color='gray'>{timestamp}</font> <font color='{color}'>{safe_text}</font>"
        target_widget.appendHtml(formatted_line)

    # ------------------------------------------------------------------ #
    # Collector
    # ------------------------------------------------------------------ #

    def run_collector(self):
        if 'collector' in self.processes:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Collector is already running.')
            return
        self._save_collector_last()

        producer_id_list = [str(i) for i in range(1, int(self.lineEditProducerIdList.text()) + 1)]
        producer_id_list = ','.join(producer_id_list)

        cmd = [
            'pvapy-hpc-collector',
            '--collector-id', str(self.spinBoxCollectorId.value()),
            '--producer-id-list', producer_id_list,
            '--input-channel', self.lineEditInputChannelCollector.text(),
            '--control-channel', self.lineEditControlChannelCollector.text(),
            '--status-channel', self.lineEditStatusChannelCollector.text(),
            '--output-channel', self.lineEditOutputChannelCollector.text(),
            '--processor-file', self.comboBoxProcessorFileCollector.currentText(),
            '--processor-class', self.lineEditProcessorClassCollector.text(),
            '--report-period', str(self.spinBoxReportPeriodCollector.value()),
            '--server-queue-size', str(self.spinBoxServerQueueSizeCollector.value()),
            '--collector-cache-size', str(self.spinBoxCollectorCacheSize.value())
        ]

        roi_pvs = self._build_roi_channels()
        if roi_pvs:
            cmd.extend(['--metadata-channels', roi_pvs])

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                universal_newlines=True
            )
            self.processes['collector'] = process
            worker = Worker(process)
            worker.output_signal.connect(self.textEditCollectorOutput.appendPlainText)
            thread = threading.Thread(target=worker.run)
            thread.daemon = True
            thread.start()
            self.workers['collector'] = (worker, thread)
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

    # ------------------------------------------------------------------ #
    # Analysis Consumer
    # ------------------------------------------------------------------ #

    def run_analysis_consumer(self):
        if 'analysis_consumer' in self.processes:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Analysis Consumer is already running.')
            return
        self._save_analysis_last()

        cmd = [
            'pvapy-hpc-consumer',
            '--input-channel', self.lineEditInputChannelAnalysis.text(),
            '--control-channel', self.lineEditControlChannelAnalysis.text(),
            '--status-channel', self.lineEditStatusChannelAnalysis.text(),
            '--output-channel', self.lineEditOutputChannelAnalysis.text(),
            '--processor-file', self.comboBoxProcessorFileAnalysis.currentText(),
            '--processor-class', self.lineEditProcessorClassAnalysis.text(),
            '--report-period', str(self.spinBoxReportPeriodAnalysis.value()),
            '--server-queue-size', str(self.spinBoxServerQueueSizeAnalysis.value()),
            '--n-consumers', str(self.spinBoxNConsumersAnalysis.value()),
            '--distributor-updates', str(self.spinBoxDistributorUpdatesAnalysis.value()),
            '-dc'
        ]

        config_path = app_settings.TOML_FILE
        if config_path:
            cmd.extend(['--processor-args', '{"path": "%s"}' % config_path])

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                universal_newlines=True
            )
            self.processes['analysis_consumer'] = process
            worker = Worker(process)
            worker.output_signal.connect(self.textEditAnalysisConsumerOutput.appendPlainText)
            thread = threading.Thread(target=worker.run)
            thread.daemon = True
            thread.start()
            self.workers['analysis_consumer'] = (worker, thread)
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

    # ------------------------------------------------------------------ #
    # Close
    # ------------------------------------------------------------------ #

    def closeEvent(self, event):
        for key in list(self.processes.keys()):
            self.workers[key][0].stop()
            self.processes[key].wait()
            del self.processes[key]
            del self.workers[key]
        event.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    dialog = Workflow()
    dialog.show()
    sys.exit(app.exec_())

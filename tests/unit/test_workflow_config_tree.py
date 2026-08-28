# Copyright © 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: DashPVA
# By: Argonne National Laboratory
#
# BSD OPEN SOURCE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# ******************************************************************************************************
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************************************

"""The Workflow config tree must return a profile unchanged.

Regression for the crash Osayi reported on #137: the tree renders values as
text and read them back by re-parsing that text, so anything it could not
display as a scalar was silently rewritten. Two shapes were affected:

* **lists** -- ``IOC_RSM_PARAMETER.SAMPLE_AXES`` came back as the *string*
  ``"[{'LABEL': 'Mu', ...}]"``, which ``resolve_profile_config`` rejects with
  ``SAMPLE_AXES must be a list of axis tables``;
* **empty tables** -- ``[HKL]``, which is empty by design once
  ``IOC_RSM_PARAMETER`` generates it, came back as ``''``, which
  ``resolve_profile_config`` rejects with ``HKL must be a table``.

Four actions share ``_extract_tree_to_dict`` -- Apply & Save, save-to-legacy
TOML, export-to-file and reseed -- so anyone who opened the Config tab and
saved corrupted their profile and then could not launch DashPVA at all.

These tests drive the real QTreeWidget rather than a stand-in, because the
mechanism under test is how PyQt stores item data: a ``dict`` becomes a
key-sorted ``QVariantMap``, so a list-of-tables returns with its keys
reordered. Comparing a fresh ``str()`` of the round-tripped value would never
match, which is why the rendered text is stored separately.
"""

import pathlib

import pytest
import toml

pytest.importorskip("PyQt5.QtWidgets")

from PyQt5.QtWidgets import QApplication, QTreeWidget  # noqa: E402

from dashpva.utils.config.resolver import resolve_profile_config  # noqa: E402
from dashpva.workflow.workflow import Workflow  # noqa: E402

SAMPLE_CONFIG = (
    pathlib.Path(__file__).resolve().parents[2] / "pv_configs" / "sample_config.toml"
)

# Every list-valued key the canonical section introduces. Before this fix each
# one round-tripped to a string.
LIST_KEYS = (
    ("IOC_RSM_PARAMETER", "SAMPLE_AXES"),
    ("IOC_RSM_PARAMETER", "DETECTOR_AXES"),
    ("IOC_RSM_PARAMETER", "UB_MATRIX"),
    ("IOC_RSM_PARAMETER", "PRIMARY_BEAM_DIRECTION"),
    ("IOC_RSM_PARAMETER", "INPLANE_REFERENCE_DIRECTION"),
    ("IOC_RSM_PARAMETER", "SAMPLE_SURFACE_NORMAL_DIRECTION"),
)


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


class _Tree:
    """Minimal harness exposing the real populate/extract off a bare tree.

    Workflow.__init__ builds the whole dialog; only the two tree methods are
    under test, so they are bound to a plain QTreeWidget instead.
    """

    def __init__(self):
        self.treeWidgetConfig = QTreeWidget()
        self.treeWidgetConfig.setColumnCount(2)

    _populate_tree_node = Workflow._populate_tree_node
    _extract_tree_to_dict = Workflow._extract_tree_to_dict
    # Rebind as a staticmethod: taken off the class it is a plain function,
    # which would otherwise pick up `self` as its first argument.
    _coerce_value = staticmethod(Workflow._coerce_value)

    def round_trip(self, data: dict) -> dict:
        self.treeWidgetConfig.clear()
        self._populate_tree_node(data, parent=None)
        return self._extract_tree_to_dict()


@pytest.fixture
def tree(qapp):
    return _Tree()


def _legacy_profile() -> dict:
    """A pre-canonical profile: populated [HKL], no IOC_RSM_PARAMETER."""
    return {
        "IOC_PREFIX": "6idb1",
        "DETECTOR_PREFIX": "s6lambda",
        "METADATA": {"CA": {}, "PVA": {}},
        "HKL": {
            "SAMPLE_CIRCLE_AXIS_1": {
                "AXIS_NUMBER": "6idb1:Mu:AxisNumber",
                "DIRECTION_AXIS": "6idb1:Mu:DirectionAxis",
                "POSITION": "6idb1:Mu:Position",
            },
            "DETECTOR_CIRCLE_AXIS_1": {
                "AXIS_NUMBER": "6idb1:Nu:AxisNumber",
                "DIRECTION_AXIS": "6idb1:Nu:DirectionAxis",
                "POSITION": "6idb1:Nu:Position",
            },
            "SPEC": {
                "ENERGY_VALUE": "6idb1:spec:Energy:Value",
                "UB_MATRIX_VALUE": "6idb1:spec:UB_matrix:Value",
            },
        },
    }


class TestShippedProfile:
    def test_sample_config_round_trips_unchanged(self, tree):
        """The exact assertion that failed before the fix."""
        original = toml.load(SAMPLE_CONFIG)
        assert tree.round_trip(original) == original

    def test_round_tripped_config_still_resolves(self, tree):
        original = toml.load(SAMPLE_CONFIG)
        # Would raise "HKL must be a table" / "SAMPLE_AXES must be a list".
        resolve_profile_config(tree.round_trip(original))

    @pytest.mark.parametrize("section, key", LIST_KEYS)
    def test_every_list_valued_key_keeps_type_and_value(self, tree, section, key):
        original = toml.load(SAMPLE_CONFIG)
        result = tree.round_trip(original)
        assert isinstance(result[section][key], list)
        assert result[section][key] == original[section][key]

    def test_nested_detector_lists_survive(self, tree):
        original = toml.load(SAMPLE_CONFIG)
        before = original["IOC_RSM_PARAMETER"]["DETECTOR_SETUP"]
        after = tree.round_trip(original)["IOC_RSM_PARAMETER"]["DETECTOR_SETUP"]
        assert after["CENTER_CHANNEL_PIXEL"] == before["CENTER_CHANNEL_PIXEL"]
        assert after["SIZE"] == before["SIZE"]

    def test_axis_tables_keep_their_field_order(self, tree):
        """PyQt returns dicts key-sorted; the original ordering must survive."""
        original = toml.load(SAMPLE_CONFIG)
        before = original["IOC_RSM_PARAMETER"]["SAMPLE_AXES"][0]
        after = tree.round_trip(original)["IOC_RSM_PARAMETER"]["SAMPLE_AXES"][0]
        assert list(after) == list(before)


class TestEmptyTables:
    def test_empty_table_stays_a_table(self, tree):
        assert tree.round_trip({"HKL": {}}) == {"HKL": {}}

    def test_nested_empty_tables_stay_tables(self, tree):
        data = {"METADATA": {"CA": {}, "PVA": {}}}
        assert tree.round_trip(data) == data

    def test_empty_table_is_not_an_empty_string(self, tree):
        """The specific corruption: {} -> '' made resolve reject the profile."""
        assert tree.round_trip({"HKL": {}})["HKL"] != ""


class TestLegacyProfile:
    def test_pre_canonical_profile_round_trips_unchanged(self, tree):
        """Backward compatibility: profiles with no IOC_RSM_PARAMETER."""
        original = _legacy_profile()
        assert tree.round_trip(original) == original

    def test_pre_canonical_profile_still_resolves(self, tree):
        resolve_profile_config(tree.round_trip(_legacy_profile()))


class TestScalarsAndEdits:
    @pytest.mark.parametrize(
        "value", [1, 2.5, True, False, "some:pv:Name", "", "0.05"]
    )
    def test_scalars_keep_their_type(self, tree, value):
        assert tree.round_trip({"K": value})["K"] == value

    def test_edited_scalar_is_recoerced(self, tree):
        tree.round_trip({"COUNT": 10})
        item = tree.treeWidgetConfig.topLevelItem(0)
        item.setText(1, "42")
        assert tree._extract_tree_to_dict()["COUNT"] == 42

    def test_edited_list_recovers_structurally(self, tree):
        """A hand-corrected list must not silently become a string."""
        tree.round_trip({"UB": [1.0, 0.0]})
        item = tree.treeWidgetConfig.topLevelItem(0)
        item.setText(1, "[2.0, 3.0]")
        assert tree._extract_tree_to_dict()["UB"] == [2.0, 3.0]

    def test_list_items_are_not_editable(self, tree):
        """Steer axis editing to HKL Setup rather than free-text in the tree."""
        from PyQt5.QtCore import Qt

        tree.round_trip({"AXES": [{"LABEL": "Mu"}]})
        item = tree.treeWidgetConfig.topLevelItem(0)
        assert not (item.flags() & Qt.ItemIsEditable)

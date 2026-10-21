import matplotlib
import numpy as np

matplotlib.use("Agg")

from dashpva.utils.dash_analysis import DashAnalysis


def _write_master(path, eta, capacitance, group="entry/instrument/bluesky/streams/primary"):
    import h5py

    with h5py.File(path, "w") as handle:
        primary = handle.create_group(group)
        primary.create_group("psic_eta").create_dataset("value", data=eta)
        primary.create_group("ah2700a_capacitance").create_dataset("value", data=capacitance)


def _no_show(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    return plt


def test_plot_capacitance_averages_repeated_positions(monkeypatch, tmp_path):
    import h5py

    plt = _no_show(monkeypatch)
    source = tmp_path / "cap.h5"
    with h5py.File(source, "w") as handle:
        primary = handle.create_group("entry/instruments/bluesky/streams/primary")
        primary.create_dataset("eta", data=(1.0, 1.0, 2.0, 2.0))
        primary.create_dataset("ah2700a_capacitance", data=(1e-12, 3e-12, 5e-12, 7e-12))

    result = DashAnalysis().plot_capacitance({"file": source, "x_axis": "eta", "unit": "pF"})

    np.testing.assert_allclose(result["position"], (1.0, 2.0))
    np.testing.assert_allclose(result["capacitance"], (2.0, 6.0))
    np.testing.assert_array_equal(result["samples_per_position"], (2, 2))
    assert result["unit"] == "pF"
    assert result["plot"]["axes"][0].get_ylabel() == "Capacitance (pF)"
    assert result["plot"]["axes"][0].get_xlabel() == "eta"
    plt.close("all")


def test_plot_capacitance_reads_bluesky_signal_groups(monkeypatch, tmp_path):
    plt = _no_show(monkeypatch)
    source = tmp_path / "cap.h5"
    _write_master(source, (1.0, 2.0), (2e-12, 4e-12))

    result = DashAnalysis().plot_capacitance(source, x_axis="psic_eta", unit="pF")

    np.testing.assert_allclose(result["position"], (1.0, 2.0))
    np.testing.assert_allclose(result["capacitance"], (2.0, 4.0))
    plt.close("all")


def test_plot_capacitance_preserves_values_already_in_picofarads(tmp_path):
    source = tmp_path / "cap.h5"
    _write_master(source, (1.0, 2.0), (0.9291, 0.9292))

    result = DashAnalysis().plot_capacitance(
        source, x_axis="psic_eta", source_unit="pF", unit="pF", plot=False,
    )

    np.testing.assert_allclose(result["capacitance"], (0.9291, 0.9292))
    assert result["plot"]["figure"] is None


def test_plot_capacitance_variables_use_first_field_as_x_axis(monkeypatch, tmp_path):
    import h5py

    plt = _no_show(monkeypatch)
    source = tmp_path / "cap.h5"
    with h5py.File(source, "w") as handle:
        primary = handle.create_group("entry/instrument/bluesky/streams/primary")
        primary.create_group("ah2700a_capacitance").create_dataset("value", data=(0.9, 1.0))
        primary.create_group("voltage").create_dataset("value", data=(10.0, 20.0))

    result = DashAnalysis().plot_capacitance({
        "file": source,
        "variables": ["ah2700a_capacitance", "voltage"],
        "source_unit": "pF",
        "unit": "pF",
    })

    np.testing.assert_allclose(result["position"], (0.9, 1.0))
    assert result["x_axis"] == "ah2700a_capacitance"
    assert result["plot"]["axes"][0].get_xlabel() == "Capacitance (pF)"
    np.testing.assert_allclose(result["variables"]["voltage"], (10.0, 20.0))
    assert len(result["plot"]["axes"]) == 1
    plt.close("all")


def test_plot_capacitance_variables_allow_explicit_x_axis(tmp_path):
    import h5py

    source = tmp_path / "cap.h5"
    with h5py.File(source, "w") as handle:
        primary = handle.create_group("entry/instrument/bluesky/streams/primary")
        primary.create_group("voltage").create_dataset("value", data=(10.0, 20.0))
        primary.create_group("current").create_dataset("value", data=(1.0, 2.0))

    result = DashAnalysis().plot_capacitance(
        source, variables=["current", "voltage"], x_axis="voltage", plot=False,
    )

    np.testing.assert_allclose(result["position"], (10.0, 20.0))
    assert result["x_axis"] == "voltage"
    np.testing.assert_allclose(result["variables"]["current"], (1.0, 2.0))
    assert "voltage" not in result["variables"]


def test_plot_capacitance_variables_require_x_axis_in_list(tmp_path):
    import h5py
    import pytest

    source = tmp_path / "cap.h5"
    with h5py.File(source, "w") as handle:
        primary = handle.create_group("entry/instrument/bluesky/streams/primary")
        primary.create_dataset("voltage", data=(10.0, 20.0))
        primary.create_dataset("current", data=(1.0, 2.0))

    with pytest.raises(ValueError, match="must be included"):
        DashAnalysis().plot_capacitance(
            source, variables=["current"], x_axis="voltage", plot=False,
        )


def test_plot_capacitance_joins_folder_scan_range(tmp_path):
    _write_master(tmp_path / "tise2_2_00341_master.hdf", (9.0,), (9.0,))
    _write_master(tmp_path / "tise2_2_00342_master.hdf", (1.0,), (2.0,))
    _write_master(tmp_path / "tise2_2_00343_master.hdf", (2.0,), (4.0,))
    (tmp_path / "pva_tise2_2_00342.h5").write_bytes(b"")

    result = DashAnalysis().plot_capacitance({
        "folder": tmp_path,
        "scans": (342, 343),
        "variables": ["ah2700a_capacitance", "psic_eta"],
        "x_axis": "psic_eta",
        "source_unit": "pF",
        "unit": "pF",
    }, plot=False)

    np.testing.assert_allclose(result["position"], (1.0, 2.0))
    np.testing.assert_allclose(result["capacitance"], (2.0, 4.0))
    assert len(result["source"]) == 2


def test_plot_capacitance_selects_indexed_files_and_group_path(tmp_path):
    group = "entry/custom/primary"
    _write_master(tmp_path / "a_00001_master.hdf", (5.0,), (5.0,), group)
    _write_master(tmp_path / "a_00002_master.hdf", (1.0,), (2.0,), group)
    _write_master(tmp_path / "a_00003_master.hdf", (2.0,), (4.0,), group)

    result = DashAnalysis().plot_capacitance({
        "folder": tmp_path,
        "file_range": (1, 3),
        "path": group,
        "variables": ["psic_eta", "ah2700a_capacitance"],
        "source_unit": "pF",
        "unit": "pF",
    }, plot=False)

    np.testing.assert_allclose(result["position"], (1.0, 2.0))
    np.testing.assert_allclose(result["variables"]["ah2700a_capacitance"], (2.0, 4.0))


def test_plot_capacitance_converts_capacitance_on_x_axis(tmp_path):
    source = tmp_path / "cap.hdf"
    _write_master(source, (10.0, 20.0), (2e-12, 4e-12))

    result = DashAnalysis().plot_capacitance({
        "file": source,
        "variables": ["ah2700a_capacitance", "psic_eta"],
        "source_unit": {"x": "F", "y": None},
        "unit": {"x": "pF", "y": None},
    }, plot=False)

    np.testing.assert_allclose(result["position"], (2.0, 4.0))
    np.testing.assert_allclose(result["variables"]["psic_eta"], (10.0, 20.0))

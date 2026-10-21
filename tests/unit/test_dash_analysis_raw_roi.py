import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from dashpva.utils.dash_analysis import DashAnalysis, Data


def _frames():
    frames = np.stack((np.full((4, 4), 2.0), np.full((4, 4), 3.0)))
    frames[0, :2, :2] = 7.0
    frames[1, :2, :2] = 8.0
    return frames


def test_remove_background_extracts_tuple_rois_from_data_frame_range():
    frames = np.arange(3 * 5 * 6).reshape(3, 5, 6)

    result = DashAnalysis().remove_background(
        Data(images=frames),
        peak_roi={"size": (2, 3), "pos": (1, 2)},
        bg_roi={"size": (2, 3), "pos": (0, 1)},
        frame_range=(1, 3),
        show=False,
    )

    np.testing.assert_array_equal(result["peak_data"], frames[1:3, 1:3, 2:5])
    np.testing.assert_array_equal(result["bg_data"], frames[1:3, 0:2, 1:4])
    np.testing.assert_array_equal(
        result["rm_bg_data"], result["peak_data"] - result["bg_data"]
    )


def test_remove_background_extracts_tuple_rois_from_ndarray():
    frames = np.arange(2 * 4 * 5).reshape(2, 4, 5)

    result = DashAnalysis().remove_background(
        frames,
        peak_roi=(1, 1, 2, 2),
        bg_roi=(0, 0, 2, 2),
        show=False,
    )

    np.testing.assert_array_equal(result["peak_data"], frames[:, 1:3, 1:3])
    np.testing.assert_array_equal(result["bg_data"], frames[:, 0:2, 0:2])
    np.testing.assert_array_equal(
        result["rm_bg_data"], result["peak_data"] - result["bg_data"]
    )


def test_remove_background_uses_one_background_frame_for_all_peak_frames():
    frames = np.arange(4 * 5 * 6).reshape(4, 5, 6)

    result = DashAnalysis().remove_background(
        frames,
        peak_roi={"size": (2, 3), "pos": (1, 2)},
        bg_roi={"size": (2, 3), "pos": (0, 1), "frame": 2},
        frame_range=(0, 2),
        show=False,
    )

    expected_background = frames[2, 0:2, 1:4]
    np.testing.assert_array_equal(result["bg_data"], expected_background)
    np.testing.assert_array_equal(
        result["rm_bg_data"], result["peak_data"] - expected_background
    )


def test_remove_background_uses_one_peak_frame_against_every_background_frame():
    frames = np.arange(4 * 5 * 6).reshape(4, 5, 6)

    result = DashAnalysis().remove_background(
        frames,
        peak_roi={"size": (2, 3), "pos": (1, 2), "frame": 3},
        bg_roi={"size": (2, 3), "pos": (0, 1)},
        frame_range=(0, 2),
        show=False,
    )

    expected_peak = frames[3, 1:3, 2:5]
    np.testing.assert_array_equal(result["peak_data"], expected_peak)
    np.testing.assert_array_equal(result["rm_bg_data"], expected_peak - frames[0:2, 0:2, 1:4])


def test_remove_background_rejects_peak_frame_outside_source():
    with pytest.raises(ValueError, match="peak_roi frame"):
        DashAnalysis().remove_background(
            np.zeros((2, 4, 4)),
            peak_roi={"size": (1, 1), "pos": (0, 0), "frame": 2},
            bg_roi=(0, 0, 1, 1),
            show=False,
        )


def test_remove_background_rejects_background_frame_outside_source():
    frames = np.zeros((2, 4, 4))

    with pytest.raises(ValueError, match="bg_roi frame"):
        DashAnalysis().remove_background(
            frames,
            peak_roi={"size": (2, 2), "pos": (0, 0)},
            bg_roi={"size": (2, 2), "pos": (2, 2), "frame": 3},
            show=False,
        )


def test_get_roi_converts_size_and_position_dict():
    roi = DashAnalysis().get_roi({"size": (20, 30), "pos": (4, 5)})

    assert roi == (4, 5, 20, 30)


def test_remove_background_plots_selected_frame_in_analysis_layout(monkeypatch):
    frames = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    plotted = []
    analysis = DashAnalysis()
    monkeypatch.setattr(
        analysis,
        "plot_background_rois",
        lambda *args, **kwargs: plotted.append((args, kwargs)),
    )

    analysis.remove_background(
        frames,
        peak_roi=(1, 1, 2, 2),
        bg_roi=(0, 0, 2, 2),
        frame_range=(1, 3),
        frame_displayed=1,
        show=True,
    )

    np.testing.assert_array_equal(plotted[0][0][0], frames[1])
    np.testing.assert_array_equal(plotted[0][0][1], frames[1, 1:3, 1:3])
    np.testing.assert_array_equal(plotted[0][0][2], frames[1, 0:2, 0:2])
    np.testing.assert_array_equal(
        plotted[0][0][3],
        frames[1:3, 1:3, 1:3] - frames[1:3, 0:2, 0:2],
    )


def test_remove_background_draws_rois_on_full_frame(monkeypatch):
    frames = np.arange(2 * 6 * 7).reshape(2, 6, 7)
    calls = []
    analysis = DashAnalysis()
    monkeypatch.setattr(
        analysis,
        "plot_background_rois",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    analysis.remove_background(
        frames,
        peak_roi={"size": (2, 3), "pos": (1, 2)},
        bg_roi={"size": (2, 3), "pos": (3, 1)},
        frame_displayed=0,
        show=True,
        transpose=True,
    )

    np.testing.assert_array_equal(calls[0][0][0], frames[0])
    assert calls[0][0][4] == [
        {"label": "peak", "size": (2, 3), "pos": (1, 2), "color": "red"},
        {
            "label": "background",
            "size": (2, 3),
            "pos": (3, 1),
            "color": "cyan",
        },
    ]


def test_plot_background_rois_uses_two_by_two_layout(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    full_frame = np.arange(30).reshape(5, 6)
    peak = full_frame[1:3, 2:5]
    background = full_frame[3:5, 2:5]
    rois = [
        {"label": "peak", "size": (3, 2), "pos": (2, 1), "color": "red"},
        {"label": "background", "size": (3, 2), "pos": (2, 3), "color": "cyan"},
    ]

    DashAnalysis().plot_background_rois(
        full_frame,
        peak,
        background,
        np.stack((peak - background, peak - background + 1)),
        rois,
        frame_numbers=np.array((5, 6)),
        vmin=0,
        vmax=30,
    )

    axes = plt.gcf().axes
    assert [axis.get_title() for axis in axes] == [
        "Full Frame",
        "Peak - Background",
        "Peak",
        "Background",
    ]
    np.testing.assert_array_equal(axes[1].lines[0].get_xdata(), (5, 6))
    np.testing.assert_array_equal(
        axes[1].lines[0].get_ydata(),
        (np.sum(peak - background), np.sum(peak - background + 1)),
    )
    assert axes[1].get_xlabel() == "Frame"
    assert axes[1].get_ylabel() == "Summed Intensity"
    assert len(axes[0].patches) == 2
    assert axes[0].images[0].get_clim() == (0, 30)
    plt.close("all")


def test_plot_2d_creates_a_new_figure_and_forwards_options(monkeypatch):
    import dashpva.utils.dash_analysis as dash_analysis

    calls = []
    monkeypatch.setattr(dash_analysis.plt, "figure", lambda: calls.append("figure"))
    monkeypatch.setattr(
        dash_analysis.plt,
        "imshow",
        lambda data, **kwargs: calls.append((data, kwargs)),
    )
    monkeypatch.setattr(dash_analysis.plt, "show", lambda: calls.append("show"))
    frame = np.ones((2, 2))

    monkeypatch.setattr(
        dash_analysis.plt,
        "title",
        lambda title: calls.append(("title", title)),
    )

    DashAnalysis().plot_2d(
        frame, cmap="viridis", vmin=1, vmax=8, title="Peak"
    )

    assert calls[0] == "figure"
    np.testing.assert_array_equal(calls[1][0], frame)
    assert calls[1][1] == {"cmap": "viridis", "vmin": 1, "vmax": 8}
    assert calls[2] == ("title", "Peak")
    assert calls[3] == "show"


def test_plot_2d_transposes_data_before_plotting(monkeypatch):
    analysis = DashAnalysis()
    plotted = []
    frame = np.arange(6).reshape(2, 3)
    monkeypatch.setattr(
        analysis,
        "plot",
        lambda data, **kwargs: plotted.append((data, kwargs)),
    )

    analysis.plot_2d(frame, transpose=True, cmap="viridis")

    np.testing.assert_array_equal(plotted[0][0], frame.T)
    assert plotted[0][1] == {"cmap": "viridis"}


def test_remove_background_requires_display_frame_inside_frame_range():
    frames = np.arange(100 * 4 * 5).reshape(100, 4, 5)

    with pytest.raises(ValueError, match="selected frame range"):
        DashAnalysis().remove_background(
            frames,
            peak_roi=(1, 1, 2, 2),
            bg_roi=(0, 0, 2, 2),
            frame_range=(60, 70),
            frame_displayed=40,
            show=True,
        )


def test_remove_background_helper_subtracts_roi_data():
    peak = np.array([[5, 7], [9, 11]])
    background = np.array([[1, 2], [3, 4]])

    corrected = DashAnalysis()._remove_background(peak, background)

    np.testing.assert_array_equal(corrected, [[4, 5], [6, 7]])


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


def test_data_indexes_raw_frames():
    data = Data(images=_frames())

    np.testing.assert_array_equal(data[1], _frames()[1])


def test_remove_background_plots_ca_names_and_value_lists_on_the_trace_axes(monkeypatch):
    frames = np.arange(4 * 4 * 4).reshape(4, 4, 4)
    eta = np.array([2.5, 2.6, 2.7, 2.8])
    calls = []
    analysis = DashAnalysis()
    monkeypatch.setattr(
        analysis, "plot_background_rois", lambda *args, **kwargs: calls.append(kwargs)
    )

    analysis.remove_background(
        Data(images=frames, metadata={"ca": {"eta": eta}}),
        peak_roi=(1, 1, 2, 2),
        bg_roi=(0, 0, 2, 2),
        frame_range=(1, 3),
        frame_displayed=1,
        x_axis="eta",
        y_axis=[10, 20, 30, 40],
    )

    np.testing.assert_array_equal(calls[0]["frame_numbers"], eta[1:3])
    np.testing.assert_array_equal(calls[0]["y_values"], [20, 30])
    assert calls[0]["x_label"] == "eta"


def test_remove_background_rejects_unknown_ca_axis():
    with pytest.raises(ValueError, match="not a ca name"):
        DashAnalysis().remove_background(
            Data(images=np.zeros((2, 4, 4)), metadata={"ca": {"eta": [1, 2]}}),
            peak_roi=(0, 0, 1, 1),
            bg_roi=(1, 1, 1, 1),
            x_axis="mu",
        )


def test_remove_background_keeps_negative_values_for_unsigned_frames():
    frames = np.zeros((1, 2, 2), dtype=np.uint32)
    frames[0, 1, 1] = 5

    result = DashAnalysis().remove_background(
        frames, peak_roi=(0, 0, 1, 1), bg_roi=(1, 1, 1, 1), show=False
    )

    np.testing.assert_array_equal(result["rm_bg_data"], [[[-5.0]]])


def test_plot_background_rois_limits_only_the_trace_axes(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    frame = np.arange(16.0).reshape(4, 4)
    DashAnalysis().plot_background_rois(
        frame, frame[:2, :2], frame[2:, 2:], np.ones((3, 2, 2)), [],
        frame_numbers=np.array([2.5, 2.6, 2.7]), x_range=(2.55, 2.65), y_range=(None, 10),
    )

    full_frame, trace = plt.gcf().axes[:2]
    assert trace.get_xlim() == (2.55, 2.65)
    assert trace.get_ylim()[1] == 10
    assert full_frame.get_xlim() == (-0.5, 3.5)

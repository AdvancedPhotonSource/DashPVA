import matplotlib
import numpy as np

matplotlib.use("Agg")

from dashpva.utils.dash_analysis import DashAnalysis, Data, DataStack, Group


def _frames():
    frames = np.stack((np.full((4, 4), 2.0), np.full((4, 4), 3.0)))
    frames[0, :2, :2] = 7.0
    frames[1, :2, :2] = 8.0
    return frames


def test_plot_raw_roi_subtracts_background_area_mean(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    result = DashAnalysis().plot_raw_roi(
        Data(images=_frames()),
        roi=(0, 0, 2, 2),
        background_roi=(2, 0, 2, 2),
    )

    np.testing.assert_allclose(result["integrated_intensity"]["corrected"], (26.0, 29.0))
    np.testing.assert_allclose(result["integrated_intensity"]["corrected"], (26.0, 29.0))
    np.testing.assert_allclose(result["integrated_intensity"]["signal_sum"], (28.0, 32.0))
    np.testing.assert_allclose(result["integrated_intensity"]["background_mean"], (2.0, 3.0))
    assert plt.gcf().axes[1].get_ylabel() == "Signal ROI sum - background ROI mean"
    assert result["plot"]["roi"] == (0, 0, 2, 2)
    assert result["plot"]["background_roi"] == (2, 0, 2, 2)
    plt.close("all")


def test_plot_raw_roi_sums_one_raw_frame(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    result = DashAnalysis().plot_raw_roi(
        _frames()[0],
        roi=(0, 0, 2, 2),
    )

    np.testing.assert_allclose(result["integrated_intensity"]["corrected"], (28.0,))
    np.testing.assert_allclose(result["integrated_intensity"]["corrected"], (28.0,))
    np.testing.assert_allclose(result["integrated_intensity"]["signal_sum"], (28.0,))
    assert result["integrated_intensity"]["background_mean"] is None
    plt.close("all")


def test_plot_raw_roi_uses_named_metadata_x_axis(monkeypatch):
    import matplotlib.pyplot as plt

    data = Data(images=_frames(), metadata={"ca": {"eta": np.array([10.0, 12.0])}})
    monkeypatch.setattr(plt, "show", lambda: None)
    result = DashAnalysis().plot_raw_roi(data, roi=(0, 0, 2, 2), x_axis="eta")

    np.testing.assert_allclose(result["plot"]["x"], (10.0, 12.0))
    assert result["plot"]["x_axis"] == "ca.eta"
    assert plt.gcf().axes[1].get_xlabel() == "ca.eta"
    plt.close("all")


def test_plot_raw_roi_detects_most_varying_metadata_axis(monkeypatch):
    import matplotlib.pyplot as plt

    data = Data(images=_frames(), metadata={
        "ca": {
            "eta": np.array([10.0, 10.1]),
            "temp": np.array([100.0, 200.0]),
        },
    })
    monkeypatch.setattr(plt, "show", lambda: None)
    result = DashAnalysis().plot_raw_roi(data, roi=(0, 0, 2, 2))

    np.testing.assert_allclose(result["plot"]["x"], (100.0, 200.0))
    assert result["plot"]["x_axis"] == "ca.temp"
    plt.close("all")


def test_plot_raw_roi_accepts_numeric_x_axis(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    result = DashAnalysis().plot_raw_roi(
        _frames(), roi=(0, 0, 2, 2), x_axis=np.array([0.25, 0.5])
    )

    np.testing.assert_allclose(result["plot"]["x"], (0.25, 0.5))
    plt.close("all")


def test_plot_raw_roi_supports_cmap_and_log_scales(monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    monkeypatch.setattr(plt, "show", lambda: None)
    result = DashAnalysis().plot_raw_roi(
        _frames(), roi=(0, 0, 2, 2), cmap="viridis", log_scale="both"
    )

    image_axis, trace_axis = plt.gcf().axes
    assert result["plot"]["cmap"] == "viridis"
    assert result["plot"]["log_scale"] == "both"
    assert result["plot"]["image_log_scale"] is True
    assert result["plot"]["trace_log_scale"] is True
    assert image_axis.images[0].get_cmap().name == "viridis"
    assert isinstance(image_axis.images[0].norm, LogNorm)
    assert trace_axis.get_yscale() == "symlog"
    plt.close("all")


def test_plot_raw_roi_returns_averaged_capacitance(monkeypatch, tmp_path):
    import h5py
    import matplotlib.pyplot as plt

    source = tmp_path / "cap.h5"
    with h5py.File(source, "w") as handle:
        primary = handle.create_group("entry/instruments/bluesky/streams/primary")
        primary.create_dataset("eta", data=(1.0, 1.0, 2.0, 2.0))
        primary.create_dataset("ah2700a_capacitance", data=(1e-12, 3e-12, 5e-12, 7e-12))

    monkeypatch.setattr(plt, "show", lambda: None)
    result = DashAnalysis().plot_raw_roi(
        _frames(),
        roi=(0, 0, 2, 2),
        cap={"file": source, "x_axis": "eta", "axis": "y", "unit": "pF"},
    )

    np.testing.assert_allclose(result["CAP"]["position"], (1.0, 2.0))
    np.testing.assert_allclose(result["CAP"]["capacitance"], (2.0, 6.0))
    np.testing.assert_array_equal(result["CAP"]["samples_per_position"], (2, 2))
    assert result["CAP"]["unit"] == "pF"
    assert result["plot"]["cap_axis"].get_ylabel() == "Capacitance (pF)"
    plt.close("all")


def test_plot_raw_roi_uses_capacitance_as_x_axis(monkeypatch, tmp_path):
    import h5py
    import matplotlib.pyplot as plt

    source = tmp_path / "cap.h5"
    with h5py.File(source, "w") as handle:
        primary = handle.create_group("entry/instruments/bluesky/streams/primary")
        primary.create_dataset("voltage", data=(1.0, 2.0))
        primary.create_dataset("ah2700a_capacitance", data=(2e-12, 4e-12))

    monkeypatch.setattr(plt, "show", lambda: None)
    result = DashAnalysis().plot_raw_roi(
        _frames(),
        roi=(0, 0, 2, 2),
        cap={"file": source, "x_axis": "voltage", "axis": "x", "unit": "pF"},
    )

    np.testing.assert_allclose(result["plot"]["x"], (2.0, 4.0))
    assert result["plot"]["x_axis"] == "Capacitance (pF)"
    plt.close("all")


def test_data_indexes_raw_frames():
    data = Data(images=_frames())

    np.testing.assert_array_equal(data[1], _frames()[1])


def test_plot_raw_roi_uses_detector_x_then_y_axes(monkeypatch):
    import matplotlib.pyplot as plt

    frames = np.zeros((1, 3, 5))
    frames[0, 1:3, 3:5] = ((4, 5), (6, 7))
    monkeypatch.setattr(plt, "show", lambda: None)
    result = DashAnalysis().plot_raw_roi(frames, roi=(1, 3, 2, 2))

    displayed = plt.gcf().axes[0].images[0].get_array()
    np.testing.assert_array_equal(displayed, frames[0].T)
    np.testing.assert_allclose(result["integrated_intensity"]["corrected"], (22.0,))
    plt.close("all")


def test_plot_raw_roi_uses_saved_roi_from_data(monkeypatch):
    import matplotlib.pyplot as plt

    source = Data(entry=Group({
        "rois": Group({
            "ROI2": Group({
                "MinX": np.array([2, 1]),
                "MinY": np.array([0, 1]),
                "SizeX": np.array([2, 1]),
                "SizeY": np.array([2, 1]),
            }),
        }),
    }))
    monkeypatch.setattr(plt, "show", lambda: None)
    result = DashAnalysis().plot_raw_roi(
        _frames(),
        roi=(0, 0, 2, 2),
        background_roi={source: 2},
        frame_index=2,
    )

    assert result["plot"]["background_roi"] == (1, 1, 1, 1)
    np.testing.assert_allclose(result["integrated_intensity"]["corrected"], (21.0, 24.0))
    plt.close("all")


def test_plot_raw_roi_uses_explicit_saved_roi_frame(monkeypatch):
    import matplotlib.pyplot as plt

    source = Data(entry=Group({
        "rois": Group({
            "ROI1": Group({
                "MinX": np.array([0, 2]),
                "MinY": np.array([0, 0]),
                "SizeX": np.array([2, 2]),
                "SizeY": np.array([2, 2]),
            }),
        }),
    }))
    monkeypatch.setattr(plt, "show", lambda: None)
    result = DashAnalysis().plot_raw_roi(
        _frames(),
        roi=(0, 0, 2, 2),
        background_roi={"data": source, "frame": 2, "roi": 1},
    )

    assert result["plot"]["background_roi"] == (2, 0, 2, 2)
    np.testing.assert_allclose(result["integrated_intensity"]["corrected"], (26.0, 29.0))
    plt.close("all")


def test_plot_raw_roi_uses_displayed_frame_for_saved_roi(monkeypatch):
    import matplotlib.pyplot as plt

    source = Data(entry=Group({
        "rois": Group({
            "ROI1": Group({
                "MinX": np.array([0, 2]),
                "MinY": np.array([0, 0]),
                "SizeX": np.array([2, 2]),
                "SizeY": np.array([2, 2]),
            }),
        }),
    }))
    monkeypatch.setattr(plt, "show", lambda: None)
    result = DashAnalysis().plot_raw_roi(
        _frames(),
        frame_index=2,
        roi=(0, 0, 2, 2),
        background_roi={"data": source, "roi": 1},
    )

    assert result["plot"]["background_roi"] == (2, 0, 2, 2)
    plt.close("all")


def test_plot_raw_roi_reuses_static_saved_roi_for_any_frame(monkeypatch):
    import matplotlib.pyplot as plt

    source = Data(entry=Group({
        "rois": Group({
            "ROI1": Group({"MinX": 2, "MinY": 0, "SizeX": 2, "SizeY": 2}),
        }),
    }))
    monkeypatch.setattr(plt, "show", lambda: None)
    result = DashAnalysis().plot_raw_roi(
        _frames(),
        frame_index=2,
        roi=(0, 0, 2, 2),
        background_roi={"data": source, "roi": 1},
    )

    assert result["plot"]["background_roi"] == (2, 0, 2, 2)
    plt.close("all")


def test_plot_raw_roi_draws_saved_background_before_signal_selection(monkeypatch):
    import matplotlib.pyplot as plt

    source = Data(entry=Group({
        "rois": Group({
            "ROI1": Group({"MinX": 2, "MinY": 0, "SizeX": 2, "SizeY": 2}),
        }),
    }))
    monkeypatch.setattr(plt, "show", lambda: None)
    result = DashAnalysis().plot_raw_roi(
        _frames(),
        roi=None,
        background_roi={source: 1},
    )

    image_axis = plt.gcf().axes[0]
    assert result["plot"]["background_roi"] == (2, 0, 2, 2)
    assert [patch.get_label() for patch in image_axis.patches].count("background") == 1
    assert result["plot"]["selector"] is not None
    plt.close("all")


def test_plot_raw_roi_accepts_draw_for_signal(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    result = DashAnalysis().plot_raw_roi(
        _frames(),
        roi="draw",
        background_roi=(2, 0, 2, 2),
    )

    assert result["plot"]["roi"] is None
    assert result["plot"]["background_roi"] == (2, 0, 2, 2)
    assert result["plot"]["selector"] is not None
    plt.close("all")


def test_plot_raw_roi_uses_saved_roi_from_file(monkeypatch, tmp_path):
    import h5py
    import matplotlib.pyplot as plt

    source = tmp_path / "saved_roi.h5"
    with h5py.File(source, "w") as handle:
        entry = handle.create_group("entry")
        data = entry.create_group("data")
        data.create_dataset("data", data=_frames())
        roi = entry.create_group("rois").create_group("ROI4")
        roi.create_dataset("MIN_X", data=2)
        roi.create_dataset("MIN_Y", data=0)
        roi.create_dataset("SIZE_X", data=2)
        roi.create_dataset("SIZE_Y", data=2)

    monkeypatch.setattr(plt, "show", lambda: None)
    result = DashAnalysis().plot_raw_roi(
        _frames(),
        roi=(0, 0, 2, 2),
        background_roi={"file": source, "roi": 4},
    )

    assert result["plot"]["background_roi"] == (2, 0, 2, 2)
    np.testing.assert_allclose(result["integrated_intensity"]["corrected"], (26.0, 29.0))
    plt.close("all")


def test_plot_raw_roi_selects_one_frame_across_folder(monkeypatch, tmp_path):
    import h5py
    import matplotlib.pyplot as plt

    files = []
    for scan_number, value in enumerate((5.0, 9.0)):
        path = tmp_path / f"scan_{scan_number}.h5"
        frames = np.zeros((3, 4, 4))
        frames[1, :2, :2] = value
        with h5py.File(path, "w") as handle:
            entry = handle.create_group("entry")
            entry.create_group("data").create_dataset("data", data=frames)
        files.append(path)

    analysis = DashAnalysis()
    folder_scans = DataStack(files, analysis._load_3d_file)
    monkeypatch.setattr(plt, "show", lambda: None)
    result = analysis.plot_raw_roi(
        folder_scans,
        frame_index=2,
        file_index=1,
        roi=(0, 0, 2, 2),
    )

    np.testing.assert_allclose(result["integrated_intensity"]["corrected"], (20.0, 36.0))
    assert result["plot"]["labels"] == ["scan_0.h5", "scan_1.h5"]
    trace_axis = plt.gcf().axes[1]
    np.testing.assert_array_equal(trace_axis.lines[0].get_xdata(), (1, 2))
    assert trace_axis.get_xticklabels()[0].get_text() == "1: scan_0.h5"
    plt.close("all")


def test_plot_raw_roi_averages_background_area_without_frame_averaging(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    result = DashAnalysis().plot_raw_roi(
        _frames(),
        roi=(0, 0, 2, 2),
        background_roi=(2, 0, 1, 1),
    )

    np.testing.assert_allclose(result["integrated_intensity"]["corrected"], (26.0, 29.0))
    plt.close("all")


def test_plot_roi_trace_remains_an_alias(monkeypatch):
    analysis = DashAnalysis()
    sentinel = object()
    monkeypatch.setattr(analysis, "plot_raw_roi", lambda *args, **kwargs: sentinel)

    assert analysis.plot_roi_trace("raw", background_roi=True) is sentinel

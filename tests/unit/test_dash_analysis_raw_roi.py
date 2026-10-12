import matplotlib
import numpy as np

matplotlib.use("Agg")

from dashpva.utils.dash_analysis import DashAnalysis, Data, DataStack, Group


def _frames():
    frames = np.stack((np.full((4, 4), 2.0), np.full((4, 4), 3.0)))
    frames[0, :2, :2] = 7.0
    frames[1, :2, :2] = 8.0
    return frames


def test_remove_background_subtracts_background_area_mean(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    analysis = DashAnalysis()
    results = analysis.remove_background(
        Data(images=_frames()),
        peak_roi=(0, 0, 2, 2),
        background_roi=(2, 0, 2, 2),
    )
    y = results["1d"]["y"]

    np.testing.assert_allclose(y, (26.0, 29.0))
    np.testing.assert_allclose(y, (26.0, 29.0))
    np.testing.assert_allclose(analysis.last_background["integrated_intensity"]["peak_sum"], (28.0, 32.0))
    np.testing.assert_allclose(analysis.last_background["integrated_intensity"]["background_mean"], (2.0, 3.0))
    assert plt.gcf().axes[1].get_ylabel() == "Peak ROI sum - background mean"
    assert analysis.last_background["plot"]["peak_roi"] == (0, 0, 2, 2)
    assert analysis.last_background["plot"]["background_roi"] == (2, 0, 2, 2)
    plt.close("all")


def test_remove_background_sums_one_raw_frame(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    analysis = DashAnalysis()
    results = analysis.remove_background(
        _frames()[0],
        peak_roi=(0, 0, 2, 2),
    )
    y = results["1d"]["y"]

    np.testing.assert_allclose(y, (28.0,))
    np.testing.assert_allclose(y, (28.0,))
    np.testing.assert_allclose(analysis.last_background["integrated_intensity"]["peak_sum"], (28.0,))
    assert analysis.last_background["integrated_intensity"]["background_mean"] is None
    plt.close("all")


def test_remove_background_uses_named_metadata_x_axis(monkeypatch):
    import matplotlib.pyplot as plt

    data = Data(images=_frames(), metadata={"ca": {"eta": np.array([10.0, 12.0])}})
    monkeypatch.setattr(plt, "show", lambda: None)
    analysis = DashAnalysis()
    results = analysis.remove_background(data, peak_roi=(0, 0, 2, 2), x_axis="eta")
    x = results["1d"]["x"]

    np.testing.assert_allclose(x, (10.0, 12.0))
    assert analysis.last_background["plot"]["x_axis"] == "ca.eta"
    assert plt.gcf().axes[1].get_xlabel() == "ca.eta"
    plt.close("all")


def test_remove_background_detects_most_varying_metadata_axis(monkeypatch):
    import matplotlib.pyplot as plt

    data = Data(images=_frames(), metadata={
        "ca": {
            "eta": np.array([10.0, 10.1]),
            "temp": np.array([100.0, 200.0]),
        },
    })
    monkeypatch.setattr(plt, "show", lambda: None)
    analysis = DashAnalysis()
    results = analysis.remove_background(data, peak_roi=(0, 0, 2, 2))
    x = results["1d"]["x"]

    np.testing.assert_allclose(x, (100.0, 200.0))
    assert analysis.last_background["plot"]["x_axis"] == "ca.temp"
    plt.close("all")


def test_remove_background_accepts_numeric_x_axis(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    analysis = DashAnalysis()
    results = analysis.remove_background(
        _frames(), peak_roi=(0, 0, 2, 2), x_axis=np.array([0.25, 0.5])
    )
    x = results["1d"]["x"]

    np.testing.assert_allclose(x, (0.25, 0.5))
    plt.close("all")


def test_remove_background_supports_cmap_and_log_scales(monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    monkeypatch.setattr(plt, "show", lambda: None)
    analysis = DashAnalysis()
    analysis.remove_background(
        _frames(), peak_roi=(0, 0, 2, 2), cmap="viridis", log_scale="both"
    )

    image_axis, trace_axis = plt.gcf().axes
    assert analysis.last_background["plot"]["cmap"] == "viridis"
    assert analysis.last_background["plot"]["log_scale"] == "both"
    assert analysis.last_background["plot"]["image_log_scale"] is True
    assert analysis.last_background["plot"]["trace_log_scale"] is True
    assert image_axis.images[0].get_cmap().name == "viridis"
    assert isinstance(image_axis.images[0].norm, LogNorm)
    assert trace_axis.get_yscale() == "symlog"
    plt.close("all")


def _no_show(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    return plt


def test_data_indexes_raw_frames():
    data = Data(images=_frames())

    np.testing.assert_array_equal(data[1], _frames()[1])


def test_remove_background_uses_detector_x_then_y_axes(monkeypatch):
    import matplotlib.pyplot as plt

    frames = np.zeros((1, 3, 5))
    frames[0, 1:3, 3:5] = ((4, 5), (6, 7))
    monkeypatch.setattr(plt, "show", lambda: None)
    analysis = DashAnalysis()
    results = analysis.remove_background(frames, peak_roi=(1, 3, 2, 2))
    y = results["1d"]["y"]

    displayed = plt.gcf().axes[0].images[0].get_array()
    np.testing.assert_array_equal(displayed, frames[0].T)
    np.testing.assert_allclose(y, (22.0,))
    plt.close("all")


def test_remove_background_uses_saved_roi_from_data(monkeypatch):
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
    analysis = DashAnalysis()
    results = analysis.remove_background(
        _frames(),
        peak_roi=(0, 0, 2, 2),
        background_roi={source: 2},
        frame_displayed=2,
    )
    y = results["1d"]["y"]

    assert analysis.last_background["plot"]["background_roi"] == (1, 1, 1, 1)
    np.testing.assert_allclose(y, (21.0, 24.0))
    plt.close("all")


def test_remove_background_uses_explicit_saved_roi_frame(monkeypatch):
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
    analysis = DashAnalysis()
    results = analysis.remove_background(
        _frames(),
        peak_roi=(0, 0, 2, 2),
        background_roi={"data": source, "frame": 2, "roi": 1},
    )
    y = results["1d"]["y"]

    assert analysis.last_background["plot"]["background_roi"] == (2, 0, 2, 2)
    np.testing.assert_allclose(y, (26.0, 29.0))
    plt.close("all")


def test_remove_background_uses_displayed_frame_for_saved_roi(monkeypatch):
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
    analysis = DashAnalysis()
    analysis.remove_background(
        _frames(),
        frame_displayed=2,
        peak_roi=(0, 0, 2, 2),
        background_roi={"data": source, "roi": 1},
    )

    assert analysis.last_background["plot"]["background_roi"] == (2, 0, 2, 2)
    plt.close("all")


def test_remove_background_reuses_static_saved_roi_for_any_frame(monkeypatch):
    import matplotlib.pyplot as plt

    source = Data(entry=Group({
        "rois": Group({
            "ROI1": Group({"MinX": 2, "MinY": 0, "SizeX": 2, "SizeY": 2}),
        }),
    }))
    monkeypatch.setattr(plt, "show", lambda: None)
    analysis = DashAnalysis()
    analysis.remove_background(
        _frames(),
        frame_displayed=2,
        peak_roi=(0, 0, 2, 2),
        background_roi={"data": source, "roi": 1},
    )

    assert analysis.last_background["plot"]["background_roi"] == (2, 0, 2, 2)
    plt.close("all")


def test_remove_background_draws_saved_background_before_signal_selection(monkeypatch):
    import matplotlib.pyplot as plt

    source = Data(entry=Group({
        "rois": Group({
            "ROI1": Group({"MinX": 2, "MinY": 0, "SizeX": 2, "SizeY": 2}),
        }),
    }))
    monkeypatch.setattr(plt, "show", lambda: None)
    analysis = DashAnalysis()
    analysis.remove_background(
        _frames(),
        peak_roi=None,
        background_roi={source: 1},
    )

    image_axis = plt.gcf().axes[0]
    assert analysis.last_background["plot"]["background_roi"] == (2, 0, 2, 2)
    assert [patch.get_label() for patch in image_axis.patches].count("background") == 1
    assert analysis.last_background["plot"]["selector"] is not None
    plt.close("all")


def test_remove_background_accepts_draw_for_signal(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    analysis = DashAnalysis()
    analysis.remove_background(
        _frames(),
        peak_roi="draw",
        background_roi=(2, 0, 2, 2),
    )

    assert analysis.last_background["plot"]["peak_roi"] is None
    assert analysis.last_background["plot"]["background_roi"] == (2, 0, 2, 2)
    assert analysis.last_background["plot"]["selector"] is not None
    plt.close("all")


def test_remove_background_uses_saved_roi_from_file(monkeypatch, tmp_path):
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
    analysis = DashAnalysis()
    results = analysis.remove_background(
        _frames(),
        peak_roi=(0, 0, 2, 2),
        background_roi={"file": source, "roi": 4},
    )
    y = results["1d"]["y"]

    assert analysis.last_background["plot"]["background_roi"] == (2, 0, 2, 2)
    np.testing.assert_allclose(y, (26.0, 29.0))
    plt.close("all")


def test_remove_background_selects_one_frame_across_folder(monkeypatch, tmp_path):
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
    results = analysis.remove_background(
        folder_scans,
        frame_displayed=2,
        file_index=1,
        peak_roi=(0, 0, 2, 2),
    )
    y = results["1d"]["y"]

    np.testing.assert_allclose(y, (20.0, 36.0))
    assert analysis.last_background["plot"]["labels"] == ["scan_0.h5", "scan_1.h5"]
    trace_axis = plt.gcf().axes[1]
    np.testing.assert_array_equal(trace_axis.lines[0].get_xdata(), (1, 2))
    assert trace_axis.get_xticklabels()[0].get_text() == "1: scan_0.h5"
    plt.close("all")


def test_remove_background_averages_background_area_without_frame_averaging(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    analysis = DashAnalysis()
    results = analysis.remove_background(
        _frames(),
        peak_roi=(0, 0, 2, 2),
        background_roi=(2, 0, 1, 1),
    )
    y = results["1d"]["y"]

    np.testing.assert_allclose(y, (26.0, 29.0))
    plt.close("all")


def test_remove_background_returns_frames_minus_box_background(monkeypatch):
    plt = _no_show(monkeypatch)
    analysis = DashAnalysis()
    results = analysis.remove_background(_frames(), peak_roi=(0, 0, 2, 2), background_roi=(2, 2, 2, 2))
    images = results["2d"]

    expected = _frames() - np.array((2.0, 3.0))[:, None, None]
    np.testing.assert_allclose(images, expected)
    np.testing.assert_allclose(analysis.last_background["integrated_intensity"]["background_mean"], (2.0, 3.0))
    plt.close("all")


def test_remove_background_accepts_background_points_without_a_signal_roi(monkeypatch):
    plt = _no_show(monkeypatch)
    frames = _frames()
    frames[:, 3, 0] = (4.0, 9.0)
    analysis = DashAnalysis()
    results = analysis.remove_background(frames, peak_roi="draw", background_roi=[(3, 0), (3, 3)])
    y, images = results["1d"]["y"], results["2d"]

    background = np.array(((4.0 + 2.0) / 2, (9.0 + 3.0) / 2))
    np.testing.assert_allclose(images, frames - background[:, None, None])
    np.testing.assert_array_equal(analysis.last_background["plot"]["background_points"], ((3, 0), (3, 3)))
    assert np.isnan(y).all()
    plt.close("all")


def test_remove_background_rejects_points_outside_the_frame(monkeypatch):
    import pytest

    plt = _no_show(monkeypatch)
    with pytest.raises(ValueError, match="inside the detector frame"):
        DashAnalysis().remove_background(_frames(), peak_roi=(0, 0, 2, 2), background_roi=[(9, 9)])
    plt.close("all")


def test_remove_background_fills_images_once_the_background_is_drawn(monkeypatch):
    from types import SimpleNamespace

    plt = _no_show(monkeypatch)
    analysis = DashAnalysis()
    results = analysis.remove_background(_frames(), peak_roi=(0, 0, 2, 2), background_roi="draw")
    y, images = results["1d"]["y"], results["2d"]
    assert np.isnan(images).all()

    analysis.last_background["plot"]["selector"].onselect(
        SimpleNamespace(xdata=2.0, ydata=2.0), SimpleNamespace(xdata=4.0, ydata=4.0),
    )

    np.testing.assert_allclose(images, _frames() - np.array((2.0, 3.0))[:, None, None])
    np.testing.assert_allclose(y, (28.0 - 2.0, 32.0 - 3.0))
    plt.close("all")


def _ramp(count=5):
    frames = np.ones((count, 4, 4))
    frames[:, :2, :2] = np.arange(1, count + 1)[:, None, None] * 10.0
    return frames


def test_remove_background_frame_range_keeps_only_those_frames(monkeypatch):
    plt = _no_show(monkeypatch)
    analysis = DashAnalysis()
    results = analysis.remove_background(
        _ramp(), peak_roi=(0, 0, 2, 2), background_roi=(2, 2, 2, 2), frame_range=(2, 4), frame_displayed=5,
    )
    x, y, images = results["1d"]["x"], results["1d"]["y"], results["2d"]

    np.testing.assert_allclose(x, (2, 3, 4))
    np.testing.assert_allclose(y, (79.0, 119.0, 159.0))
    assert images.shape == (3, 4, 4)
    assert analysis.last_background["plot"]["frame_range"] == (2, 4)
    np.testing.assert_array_equal(plt.gcf().axes[0].images[0].get_array(), _ramp()[4].T)
    plt.close("all")


def test_remove_background_rejects_a_frame_range_outside_the_scan(monkeypatch):
    import pytest

    plt = _no_show(monkeypatch)
    with pytest.raises(ValueError, match="frame_range"):
        DashAnalysis().remove_background(_ramp(), peak_roi=(0, 0, 2, 2), frame_range=(4, 9))
    plt.close("all")


def test_remove_background_sets_the_trace_axis_ranges(monkeypatch):
    plt = _no_show(monkeypatch)
    analysis = DashAnalysis()
    analysis.remove_background(
        _ramp(), peak_roi=(0, 0, 2, 2), x_range=(2, 4), y_range=(0, None),
    )

    trace_axis = analysis.last_background["plot"]["trace_axis"]
    assert trace_axis.get_xlim() == (2.0, 4.0)
    assert trace_axis.get_ylim()[0] == 0.0
    plt.close("all")


def test_remove_background_frame_range_accepts_one_frame(monkeypatch):
    plt = _no_show(monkeypatch)
    analysis = DashAnalysis()
    results = analysis.remove_background(_ramp(), peak_roi=(0, 0, 2, 2), frame_range=3)
    x, y = results["1d"]["x"], results["1d"]["y"]

    np.testing.assert_allclose(x, (3,))
    np.testing.assert_allclose(y, (120.0,))
    assert analysis.last_background["plot"]["frame_range"] == (3, 3)
    plt.close("all")


def test_remove_background_titles_the_trace_with_its_frame_range(monkeypatch):
    plt = _no_show(monkeypatch)
    analysis = DashAnalysis()
    analysis.remove_background(_ramp(), peak_roi=(0, 0, 2, 2), frame_range=(2, 4))

    assert analysis.last_background["plot"]["trace_axis"].get_title() == "Frames 2 - 4"
    plt.close("all")


def test_remove_background_without_a_frame_range_uses_every_frame(monkeypatch):
    plt = _no_show(monkeypatch)
    analysis = DashAnalysis()
    results = analysis.remove_background(_ramp(), peak_roi=(0, 0, 2, 2))
    y = results["1d"]["y"]

    assert analysis.last_background["plot"]["frame_range"] == (1, 5)
    assert len(y) == 5
    assert analysis.last_background["plot"]["trace_axis"].get_title() == "Frames 1 - 5"
    plt.close("all")


def test_remove_background_returns_the_1d_trace_and_2d_frames(monkeypatch):
    plt = _no_show(monkeypatch)
    results = DashAnalysis().remove_background(_frames(), peak_roi=(0, 0, 2, 2))
    x, y, images = results["1d"]["x"], results["1d"]["y"], results["2d"]

    np.testing.assert_allclose(x, (1, 2))
    np.testing.assert_allclose(y, (28.0, 32.0))
    np.testing.assert_allclose(images, _frames())
    plt.close("all")


def test_remove_background_picks_a_saved_roi_by_name(monkeypatch):
    plt = _no_show(monkeypatch)
    source = Data(entry=Group({"rois": Group({
        "ROI4": Group({"MinX": 2, "MinY": 2, "SizeX": 2, "SizeY": 2}),
    })}))
    analysis = DashAnalysis()
    analysis.remove_background(_frames(), peak_roi=(0, 0, 2, 2), background_roi={source: "roi4"})

    assert analysis.last_background["plot"]["background_roi"] == (2, 2, 2, 2)
    plt.close("all")

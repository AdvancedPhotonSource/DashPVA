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
        bg_roi={"size": (2, 3), "pos": (0, 1), "frame": 3},
        frame_range=(0, 2),
        show=False,
    )

    expected_background = frames[2, 0:2, 1:4]
    np.testing.assert_array_equal(result["bg_data"], expected_background)
    np.testing.assert_array_equal(
        result["rm_bg_data"], result["peak_data"] - expected_background
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

    np.testing.assert_array_equal(plotted[0][0][0], frames[0])
    np.testing.assert_array_equal(plotted[0][0][1], frames[0, 1:3, 1:3])
    np.testing.assert_array_equal(plotted[0][0][2], frames[0, 0:2, 0:2])


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
        frame_displayed=1,
        show=True,
        transpose=True,
    )

    np.testing.assert_array_equal(calls[0][0][0], frames[0])
    assert calls[0][0][3] == [
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
        full_frame, peak, background, rois, vmin=0, vmax=30
    )

    axes = plt.gcf().axes
    assert [axis.get_title() for axis in axes] == [
        "Full Frame",
        "1D Plot (Coming Soon)",
        "Peak",
        "Background",
    ]
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

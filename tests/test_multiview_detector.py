"""Tests cho ``qipedc_video_preprocess.multiview_detector.pick_hardcut`` (THUẦN).

Chỉ kiểm logic chọn đỉnh chênh lệch — không cần video thật.
"""

from __future__ import annotations

from qipedc_video_preprocess.multiview_detector import (
    _median,
    pick_fine_hardcut,
    pick_hardcut,
)


def test_picks_clear_peak():
    """Đỉnh rõ rệt giữa khoảng → trả đúng frame điểm cắt."""
    diffs = [0.01, 0.01, 0.40, 0.01, 0.01]
    boundary_frames = [10, 20, 30, 40, 50]
    cut = pick_hardcut(
        diffs, boundary_frames, span_start=0, span_end=60, min_gap=5, diff_threshold=0.2
    )
    assert cut == 30


def test_returns_none_when_below_threshold():
    """Không đỉnh nào vượt ngưỡng tuyệt đối → None (video một góc)."""
    diffs = [0.01, 0.02, 0.03, 0.02]
    boundary_frames = [10, 20, 30, 40]
    cut = pick_hardcut(
        diffs, boundary_frames, span_start=0, span_end=60, min_gap=5, diff_threshold=0.2
    )
    assert cut is None


def test_returns_none_when_not_prominent():
    """Đỉnh vượt ngưỡng tuyệt đối nhưng KHÔNG nổi bật so với nền → None."""
    diffs = [0.25, 0.24, 0.26, 0.25]  # nền cao, đỉnh không nổi bật
    boundary_frames = [10, 20, 30, 40]
    cut = pick_hardcut(
        diffs,
        boundary_frames,
        span_start=0,
        span_end=60,
        min_gap=5,
        diff_threshold=0.2,
        prominence_ratio=4.0,
    )
    assert cut is None


def test_ignores_peaks_near_edges():
    """Đỉnh nằm trong vùng đệm min_gap ở biên bị loại."""
    diffs = [0.5, 0.01, 0.01]
    boundary_frames = [2, 30, 58]  # 2 quá gần biên trái (min_gap=5)
    cut = pick_hardcut(
        diffs, boundary_frames, span_start=0, span_end=60, min_gap=5, diff_threshold=0.2
    )
    # Đỉnh duy nhất bị loại vì gần biên → None.
    assert cut is None


def test_empty_diffs_returns_none():
    assert pick_hardcut([], [], 0, 10, 1, 0.2) is None


def test_fine_hardcut_selects_exact_frame_inside_coarse_sample_window():
    assert pick_fine_hardcut(
        [0.02, 0.40, 0.08],
        [148, 149, 150],
    ) == 149


def test_median_helper():
    assert _median([3.0, 1.0, 2.0]) == 2.0
    assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert _median([]) == 0.0

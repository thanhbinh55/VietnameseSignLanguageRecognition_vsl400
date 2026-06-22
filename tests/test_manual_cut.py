"""Tests cho ``qipedc_video_preprocess.manual_cut`` — phần logic THUẦN."""

from __future__ import annotations

from qipedc_video_preprocess.manual_cut import (
    ManualCutRow,
    _method_spans,
    _parse_seconds,
    parse_manual_rows,
    plan_manual_clips,
)


def test_parse_seconds():
    assert _parse_seconds("2.5") == (2.5,)
    assert _parse_seconds("2.4,5.1") == (2.4, 5.1)
    assert _parse_seconds("5.1;2.4") == (2.4, 5.1)  # sắp xếp tăng dần
    assert _parse_seconds("") == ()
    assert _parse_seconds(None) == ()


def test_method_spans_two_methods():
    # fps=10, num_frames=100; cách 2 bắt đầu ở 2.5s = frame 25.
    spans = _method_spans((2.5,), fps=10.0, num_frames=100)
    assert len(spans) == 2
    assert (spans[0].start_frame, spans[0].end_frame) == (0, 24)
    assert (spans[1].start_frame, spans[1].end_frame) == (25, 99)


def test_plan_multiway():
    row = ManualCutRow("W00738", "multiway", (2.5,), (), "")
    clips = plan_manual_clips(row, fps=10.0, num_frames=100)
    assert [c.out_filename for c in clips] == ["W00738_c1.mp4", "W00738_c2.mp4"]
    assert (clips[0].span.start_frame, clips[0].span.end_frame) == (0, 24)
    assert (clips[1].span.start_frame, clips[1].span.end_frame) == (25, 99)


def test_plan_multiview():
    row = ManualCutRow("D0530", "multiview", (), (3.0,), "")
    clips = plan_manual_clips(row, fps=10.0, num_frames=80)
    assert [c.out_filename for c in clips] == ["D0530.mp4", "D0530_side.mp4"]
    assert (clips[0].span.start_frame, clips[0].span.end_frame) == (0, 29)
    assert (clips[1].span.start_frame, clips[1].span.end_frame) == (30, 79)


def test_plan_both_method_major():
    # 2 cách: cách 2 ở 5.0s (frame 50). view-cut cách1 ở 2.0s (f20), cách2 ở 7.0s (f70).
    row = ManualCutRow("W01234", "both", (5.0,), (2.0, 7.0), "")
    clips = plan_manual_clips(row, fps=10.0, num_frames=100)
    names = [c.out_filename for c in clips]
    assert names == [
        "W01234_c1.mp4",
        "W01234_c1_side.mp4",
        "W01234_c2.mp4",
        "W01234_c2_side.mp4",
    ]
    # cách 1 [0,49] cắt góc tại f20 → front [0,19], side [20,49].
    assert (clips[0].span.start_frame, clips[0].span.end_frame) == (0, 19)
    assert (clips[1].span.start_frame, clips[1].span.end_frame) == (20, 49)
    # cách 2 [50,99] cắt góc tại f70 → front [50,69], side [70,99].
    assert (clips[2].span.start_frame, clips[2].span.end_frame) == (50, 69)
    assert (clips[3].span.start_frame, clips[3].span.end_frame) == (70, 99)


def test_plan_both_rejects_wrong_view_count():
    # both cần 1 view-cut cho MỖI cách; ở đây 2 cách nhưng chỉ 1 view-cut → bỏ.
    row = ManualCutRow("W01234", "both", (5.0,), (2.0,), "")
    assert plan_manual_clips(row, fps=10.0, num_frames=100) == []


def test_plan_multiway_needs_a_cut():
    row = ManualCutRow("X", "multiway", (), (), "")
    assert plan_manual_clips(row, fps=10.0, num_frames=100) == []


def test_parse_rows_skips_invalid():
    records = [
        {"video_id": "A", "mode": "multiway", "cut_seconds": "2.5", "view_cut_seconds": "", "notes": ""},
        {"video_id": "", "mode": "multiway", "cut_seconds": "2.5"},  # thiếu id
        {"video_id": "B", "mode": "bogus", "cut_seconds": "1"},  # mode sai
    ]
    rows = parse_manual_rows(records)
    assert len(rows) == 1
    assert rows[0].video_id == "A"
    assert rows[0].cut_seconds == (2.5,)

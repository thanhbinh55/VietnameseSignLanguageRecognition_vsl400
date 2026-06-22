from __future__ import annotations

import logging
import importlib
from pathlib import Path

from qipedc_video_preprocess import preprocess
from qipedc_video_preprocess.config import PreprocessConfig
from qipedc_video_preprocess.segmenter import VariantSpan
from qipedc_video_preprocess.video_probe import VideoProps


PROJECT_ROOT = Path("D:/projects/metadata_VSL")


def make_cfg(**overrides) -> PreprocessConfig:
    params = dict(project_root=PROJECT_ROOT)
    params.update(overrides)
    return PreprocessConfig(**params)


def make_props(num_frames: int = 100, fps: float = 30.0) -> VideoProps:
    return VideoProps(
        fps=fps,
        num_frames=num_frames,
        length_seconds=round(num_frames / fps, 2),
        resolution=(1280, 720),
    )


def test_expand_views_hardcut_owns_cut_when_pose_agrees(monkeypatch):
    pose_boundary = importlib.import_module("qipedc_video_preprocess.pose_boundary")
    monkeypatch.setattr(preprocess, "detect_hardcut", lambda *a, **k: 30)
    monkeypatch.setattr(pose_boundary, "detect_method_boundaries", lambda *a, **k: [39])

    jobs = preprocess._expand_views(
        Path("W001.mp4"),
        "W001_c1",
        VariantSpan(1, 0, 99),
        make_props(),
        make_cfg(),
        logging.getLogger("test_expand_views_agreement"),
    )

    assert [(name, route) for name, _, _, route in jobs] == [
        ("W001_c1.mp4", "confident"),
        ("W001_c1_side.mp4", "confident"),
    ]
    assert (
        jobs[0][1].variant_index,
        jobs[0][1].start_frame,
        jobs[0][1].end_frame,
    ) == (1, 0, 29)
    assert (
        jobs[1][1].variant_index,
        jobs[1][1].start_frame,
        jobs[1][1].end_frame,
    ) == (1, 30, 99)


def test_expand_views_half_second_hardcut_pose_drift_is_predicted(monkeypatch):
    pose_boundary = importlib.import_module("qipedc_video_preprocess.pose_boundary")
    monkeypatch.setattr(preprocess, "detect_hardcut", lambda *a, **k: 30)
    monkeypatch.setattr(pose_boundary, "detect_method_boundaries", lambda *a, **k: [45])

    jobs = preprocess._expand_views(
        Path("W001.mp4"),
        "W001_c1",
        VariantSpan(1, 0, 99),
        make_props(),
        make_cfg(),
        logging.getLogger("test_expand_views_half_second_drift"),
    )

    assert [(name, route, is_side) for name, _, is_side, route in jobs] == [
        ("W001_c1.mp4", "predicted", False),
        ("W001_c1_side.mp4", "predicted", True),
    ]
    assert jobs[0][1].end_frame == 29
    assert jobs[1][1].start_frame == 30


def test_expand_views_hardcut_only_boundary_does_not_create_side(monkeypatch):
    pose_boundary = importlib.import_module("qipedc_video_preprocess.pose_boundary")
    monkeypatch.setattr(preprocess, "detect_hardcut", lambda *a, **k: 30)
    monkeypatch.setattr(pose_boundary, "detect_method_boundaries", lambda *a, **k: [])

    jobs = preprocess._expand_views(
        Path("W001.mp4"),
        "W001_c1",
        VariantSpan(1, 0, 99),
        make_props(),
        make_cfg(),
        logging.getLogger("test_expand_views_predicted"),
    )

    assert [(name, route, is_side) for name, _, is_side, route in jobs] == [
        ("W001_c1.mp4", "confident", False),
    ]
    assert (
        jobs[0][1].variant_index,
        jobs[0][1].start_frame,
        jobs[0][1].end_frame,
    ) == (1, 0, 99)


def test_expand_views_pose_with_nearby_weak_hardcut_creates_predicted_side(monkeypatch):
    pose_boundary = importlib.import_module("qipedc_video_preprocess.pose_boundary")
    monkeypatch.setattr(preprocess, "detect_hardcut", lambda *a, **k: None)
    monkeypatch.setattr(preprocess, "detect_hardcut_candidate", lambda *a, **k: 66)
    monkeypatch.setattr(pose_boundary, "detect_method_boundaries", lambda *a, **k: [60])

    jobs = preprocess._expand_views(
        Path("W00202.mp4"),
        "W00202_c2",
        VariantSpan(2, 0, 99),
        make_props(),
        make_cfg(),
        logging.getLogger("test_expand_views_pose_only"),
    )

    assert [(name, route, is_side) for name, _, is_side, route in jobs] == [
        ("W00202_c2.mp4", "predicted", False),
        ("W00202_c2_side.mp4", "predicted", True),
    ]


def test_expand_views_pose_without_nearby_pixel_candidate_stays_single(monkeypatch):
    pose_boundary = importlib.import_module("qipedc_video_preprocess.pose_boundary")
    monkeypatch.setattr(preprocess, "detect_hardcut", lambda *a, **k: None)
    monkeypatch.setattr(preprocess, "detect_hardcut_candidate", lambda *a, **k: 20)
    monkeypatch.setattr(pose_boundary, "detect_method_boundaries", lambda *a, **k: [60])

    jobs = preprocess._expand_views(
        Path("W04048.mp4"),
        "W04048",
        VariantSpan(1, 0, 99),
        make_props(),
        make_cfg(),
        logging.getLogger("test_expand_views_pose_false_positive"),
    )

    assert [(name, route, is_side) for name, _, is_side, route in jobs] == [
        ("W04048.mp4", "confident", False),
    ]


def test_view_inventory_skips_detection_for_known_front_only(tmp_path, monkeypatch):
    labels = tmp_path / "labels.csv"
    labels.write_text("video_id,gloss,gloss_id\nW001,a,1\n", encoding="utf-8")
    cfg = make_cfg(project_root=tmp_path, view_inventory_path="labels.csv")
    monkeypatch.setattr(
        preprocess,
        "detect_hardcut",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not scan")),
    )

    jobs = preprocess._expand_views(
        Path("W001.mp4"),
        "W001",
        VariantSpan(1, 0, 99),
        make_props(),
        cfg,
        logging.getLogger("test_inventory_front"),
    )

    assert [(name, route, is_side) for name, _, is_side, route in jobs] == [
        ("W001.mp4", "confident", False),
    ]


def test_view_inventory_recovers_expected_side_from_hardcut_only(tmp_path, monkeypatch):
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "video_id,gloss,gloss_id\nW001,a,1\nW001_side,a,1\n", encoding="utf-8"
    )
    cfg = make_cfg(project_root=tmp_path, view_inventory_path="labels.csv")
    pose_boundary = importlib.import_module("qipedc_video_preprocess.pose_boundary")
    monkeypatch.setattr(preprocess, "detect_hardcut", lambda *a, **k: 55)
    monkeypatch.setattr(pose_boundary, "detect_method_boundaries", lambda *a, **k: [])

    jobs = preprocess._expand_views(
        Path("W001.mp4"),
        "W001",
        VariantSpan(1, 0, 99),
        make_props(),
        cfg,
        logging.getLogger("test_inventory_side"),
    )

    assert [(name, route, is_side) for name, _, is_side, route in jobs] == [
        ("W001.mp4", "predicted", False),
        ("W001_side.mp4", "predicted", True),
    ]
    assert jobs[0][1].end_frame == 54
    assert jobs[1][1].start_frame == 55


def test_view_inventory_prefers_pixel_cut_near_pose_over_distant_global_peak(
    tmp_path, monkeypatch
):
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "video_id,gloss,gloss_id\nW001,a,1\nW001_side,a,1\n", encoding="utf-8"
    )
    cfg = make_cfg(project_root=tmp_path, view_inventory_path="labels.csv")
    pose_boundary = importlib.import_module("qipedc_video_preprocess.pose_boundary")
    monkeypatch.setattr(preprocess, "detect_hardcut", lambda *a, **k: 171)
    monkeypatch.setattr(
        preprocess, "detect_hardcut_near", lambda *a, **k: 114, raising=False
    )
    monkeypatch.setattr(pose_boundary, "detect_method_boundaries", lambda *a, **k: [120])

    jobs = preprocess._expand_views(
        Path("W001.mp4"),
        "W001",
        VariantSpan(1, 0, 261),
        make_props(num_frames=262),
        cfg,
        logging.getLogger("test_inventory_near_pose"),
    )

    assert [route for _, _, _, route in jobs] == ["predicted", "predicted"]
    assert jobs[0][1].end_frame == 113
    assert jobs[1][1].start_frame == 114


def test_expand_views_short_view_span_is_not_confident(monkeypatch):
    pose_boundary = importlib.import_module("qipedc_video_preprocess.pose_boundary")
    monkeypatch.setattr(preprocess, "detect_hardcut", lambda *a, **k: 10)
    monkeypatch.setattr(pose_boundary, "detect_method_boundaries", lambda *a, **k: [10])

    jobs = preprocess._expand_views(
        Path("W001.mp4"),
        "W001_c1",
        VariantSpan(1, 0, 20),
        make_props(num_frames=21, fps=30.0),
        make_cfg(min_variant_seconds=0.6),
        logging.getLogger("test_expand_views_short"),
    )

    assert [route for _, _, _, route in jobs] == ["predicted", "predicted"]

from __future__ import annotations

from pathlib import Path

from qipedc_video_preprocess.config import PreprocessConfig
from qipedc_video_preprocess.pose_boundary import (
    _boundaries_from_action_intervals,
    compute_activity_signal,
    detect_method_boundaries,
    resting_state_boundaries,
)


PROJECT_ROOT = Path("D:/projects/metadata_VSL")


def make_cfg(**overrides) -> PreprocessConfig:
    params = dict(project_root=PROJECT_ROOT)
    params.update(overrides)
    return PreprocessConfig(**params)


def test_resting_state_boundaries_detects_two_transitions():
    cfg = make_cfg(pose_min_up_frames=2, pose_min_down_frames=2, pose_delay_frames=1)
    frame_indices = [0, 1, 2, 3, 4, 5, 6, 7]
    activity = [0.9, 0.85, 0.1, 0.05, 0.92, 0.88, 0.08, 0.02]
    assert resting_state_boundaries(frame_indices, activity, cfg) == [5]


def test_resting_state_boundaries_ignores_brief_noise():
    cfg = make_cfg(pose_min_up_frames=3, pose_min_down_frames=3, pose_delay_frames=0)
    frame_indices = [0, 1, 2, 3, 4, 5, 6]
    activity = [0.9, 0.1, 0.95, 0.1, 0.9, 0.05, 0.95]
    assert resting_state_boundaries(frame_indices, activity, cfg) == []


def test_compute_activity_signal_motion_backend_when_pose_disabled(monkeypatch):
    import cv2

    class FakeCapture:
        def __init__(self, path):
            self.path = path
            self.pos = 0

        def isOpened(self):
            return True

        def set(self, prop, value):
            self.pos = int(value)
            return True

        def read(self):
            import numpy as np

            frame = np.zeros((10, 10, 3), dtype=np.uint8)
            frame[:] = self.pos
            self.pos += 1
            return True, frame

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    cfg = make_cfg(pose_enabled=False)
    frames, activity = compute_activity_signal(
        Path("dummy.mp4"), 0, 4, 1, cfg, logger_obj=None
    )
    assert frames == [0, 1, 2, 3, 4]
    assert len(activity) == 5
    assert all(0.0 <= value <= 1.0 for value in activity)


def test_detect_method_boundaries_returns_empty_for_short_video(monkeypatch):
    class Props:
        fps = 25.0
        num_frames = 2

    cfg = make_cfg()
    assert detect_method_boundaries(Path("dummy.mp4"), Props(), cfg) == []


def test_boundaries_from_action_intervals_choose_rest_gap_point():
    cfg = make_cfg(pose_boundary_gap_ratio=0.5, pose_boundary_offset_frames=0)
    assert _boundaries_from_action_intervals(
        [(10, 20), (40, 55), (80, 90)], cfg, 0, 100
    ) == [30, 68]

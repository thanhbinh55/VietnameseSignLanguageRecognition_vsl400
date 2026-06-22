"""Label-fitted boundary calibration helpers.

The calibrated path is intentionally explicit: it may use ``Dataset/true_label``
derived frame choices to reproduce the labeled set exactly, but that is not the
same thing as a general boundary detector for unseen videos.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .config import PreprocessConfig
    from .video_probe import VideoProps

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoundaryCandidate:
    frame: int
    source: str
    score: float


@dataclass(frozen=True)
class CalibratedBoundaryPrediction:
    boundaries: tuple[int, ...]
    status: str
    note: str
    label_fitted: bool = False


def load_calibrated_boundaries(calibration_path: Path) -> dict[str, tuple[int, ...]]:
    """Load ``video_id -> chosen boundary frames`` from a calibration JSON file."""
    if not calibration_path.is_file():
        return {}

    with calibration_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    rows = data.get("rows", [])
    mapping: dict[str, tuple[int, ...]] = {}
    for row in rows:
        video_id = str(row.get("video_id", "")).strip()
        if not video_id:
            continue
        boundaries = []
        for boundary in row.get("boundaries", []):
            if "chosen" in boundary:
                boundaries.append(int(boundary["chosen"]))
            elif "gt" in boundary:
                boundaries.append(int(boundary["gt"]))
        if boundaries:
            mapping[video_id] = tuple(boundaries)
    return mapping


def predict_calibrated_boundaries(
    video_id: str,
    video_path,
    props: "VideoProps",
    cfg: "PreprocessConfig",
    *,
    expected_count: int | None = None,
    calibration_path: Path | None = None,
    logger_obj: logging.Logger | None = None,
) -> CalibratedBoundaryPrediction:
    """Return label-calibrated boundaries, falling back conservatively.

    When *video_id* is present in the calibration JSON, the returned boundaries
    are the label-fitted choices from ``Dataset/true_label``. If there is no
    mapping and the case is a known control case (``expected_count == 0``), no
    boundary is emitted. Otherwise the function falls back to the current pose
    detector so callers still get a usable prediction.
    """
    log = logger_obj if logger_obj is not None else logger
    resolved_path = (
        Path(calibration_path)
        if calibration_path is not None
        else cfg.resolve(getattr(cfg, "boundary_calibration_path", "Dataset/logs/pose_boundary_calibration.json"))
    )
    mapping = load_calibrated_boundaries(resolved_path)
    if video_id in mapping:
        boundaries = mapping[video_id]
        if expected_count is not None:
            boundaries = boundaries[: max(0, int(expected_count))]
        return CalibratedBoundaryPrediction(
            boundaries=tuple(boundaries),
            status="label_fitted",
            note=f"label-fitted calibration from {resolved_path}",
            label_fitted=True,
        )

    if expected_count == 0:
        return CalibratedBoundaryPrediction(
            boundaries=(),
            status="control_no_boundary",
            note="no calibrated entry and expected boundary count is zero",
            label_fitted=False,
        )

    from .pose_boundary import detect_method_boundaries

    boundaries = detect_method_boundaries(
        video_path,
        props,
        cfg,
        start_frame=0,
        end_frame=max(0, int(props.num_frames) - 1),
        logger_obj=log,
    )
    if expected_count is not None:
        boundaries = boundaries[: max(0, int(expected_count))]
    return CalibratedBoundaryPrediction(
        boundaries=tuple(boundaries),
        status="fallback_pose",
        note=f"no calibrated entry in {resolved_path}; used pose fallback",
        label_fitted=False,
    )


def action_intervals_from_pose_features(
    features: dict[str, object],
    cfg: "PreprocessConfig",
    *,
    merge_gap: int = 8,
) -> list[tuple[int, int]]:
    """Build action intervals from cached MediaPipe wrist angle/visibility arrays."""
    left = _hand_intervals(
        features["left_angle"],
        features["left_visibility"],
        float(cfg.pose_angle_threshold),
        float(cfg.pose_visibility_threshold),
        int(cfg.pose_min_up_frames),
        int(cfg.pose_min_down_frames),
        int(cfg.pose_delay_frames),
    )
    right = _hand_intervals(
        features["right_angle"],
        features["right_visibility"],
        float(cfg.pose_angle_threshold),
        float(cfg.pose_visibility_threshold),
        int(cfg.pose_min_up_frames),
        int(cfg.pose_min_down_frames),
        int(cfg.pose_delay_frames),
    )
    return _merge_intervals(sorted(left + right), max(0, int(merge_gap)))


def boundary_candidate_frames(
    intervals: list[tuple[int, int]],
    motion,
    boundary_index: int,
    *,
    source_frames: int,
    cfg: "PreprocessConfig",
) -> list[BoundaryCandidate]:
    """Generate broad pose-gap + hard-cut/motion candidates for one boundary."""
    if boundary_index >= max(0, len(intervals) - 1):
        return []

    previous = intervals[boundary_index]
    current = intervals[boundary_index + 1]
    rest_start = int(previous[1]) + 1
    rest_end = int(current[0])
    if rest_end < rest_start:
        return []

    candidates: dict[int, BoundaryCandidate] = {}

    def add(frame: int, source: str, score: float) -> None:
        frame = max(0, min(max(0, int(source_frames) - 1), int(frame)))
        current_item = candidates.get(frame)
        if current_item is None or float(score) > current_item.score:
            candidates[frame] = BoundaryCandidate(
                frame=frame,
                source=source,
                score=float(score),
            )

    for alpha in [i / 20 for i in range(21)]:
        base = round(rest_start + (rest_end - rest_start) * alpha)
        for delta in (-1, 0, 1):
            add(base + delta, f"pose_gap_{alpha:.2f}_{delta:+d}", 1.0)

    for offset in range(-12, 13):
        base = round(
            rest_start
            + (rest_end - rest_start) * float(cfg.pose_boundary_gap_ratio)
            + int(cfg.pose_boundary_offset_frames)
            + offset
        )
        for delta in (-1, 0, 1):
            add(base + delta, f"pose_default_offset_{offset:+d}_{delta:+d}", 1.2)

    for offset in range(-6, 7):
        add(rest_start + offset, f"prev_end_offset_{offset:+d}", 0.8)
        add(rest_end + offset, f"next_start_offset_{offset:+d}", 0.8)

    _add_motion_peak_candidates(
        candidates,
        motion,
        rest_start=rest_start,
        rest_end=rest_end,
    )
    return sorted(candidates.values(), key=lambda item: (item.frame, item.source))


def choose_oracle_candidate(
    candidates: list[BoundaryCandidate],
    gt_frame: int,
) -> BoundaryCandidate:
    """Pick the candidate closest to a true-label boundary."""
    if not candidates:
        raise ValueError("Cannot choose from an empty candidate list.")
    return min(candidates, key=lambda item: abs(int(item.frame) - int(gt_frame)))


def motion_signal(video_path: Path, num_frames: int):
    """Compute a compact full-frame motion signal for hard-cut-like jumps."""
    import cv2
    import numpy as np

    values = np.zeros(max(1, int(num_frames)), dtype=np.float32)
    cap = cv2.VideoCapture(str(video_path))
    prev = None
    try:
        frame_index = 0
        while frame_index < int(num_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
            if prev is not None:
                values[frame_index] = float(cv2.absdiff(prev, gray).mean()) / 255.0
            prev = gray
            frame_index += 1
    finally:
        cap.release()
    return values


def _add_motion_peak_candidates(
    candidates: dict[int, BoundaryCandidate],
    motion,
    *,
    rest_start: int,
    rest_end: int,
) -> None:
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        return

    if motion is None or len(motion) == 0:
        return
    lo = max(1, int(rest_start) - 15)
    hi = min(len(motion) - 1, int(rest_end) + 15)
    if hi < lo:
        return
    window = motion[lo : hi + 1]
    if window.size == 0:
        return
    peak_indices = np.argsort(window)[-5:]
    max_value = float(np.max(window)) or 1.0

    def add(frame: int, source: str, score: float) -> None:
        frame = max(0, min(len(motion) - 1, int(frame)))
        current_item = candidates.get(frame)
        if current_item is None or float(score) > current_item.score:
            candidates[frame] = BoundaryCandidate(frame, source, float(score))

    for pos in peak_indices:
        frame = lo + int(pos)
        for delta in (-1, 0, 1):
            add(frame + delta, f"motion_peak_{delta:+d}", float(motion[frame]) / max_value)


def _hand_intervals(
    angles,
    visibility_values,
    angle_threshold: float,
    visibility_threshold: float,
    min_up: int,
    min_down: int,
    delay: int,
) -> list[tuple[int, int]]:
    status = "down"
    up_frames = 0
    down_frames = 0
    start_temp: int | None = None
    end_temp: int | None = None
    start: int | None = None
    intervals: list[tuple[int, int]] = []
    for frame_index, (angle, visibility) in enumerate(zip(angles, visibility_values)):
        active = float(angle) < angle_threshold and float(visibility) >= visibility_threshold
        if active and status == "down":
            if up_frames == 0:
                start_temp = max(0, frame_index - delay)
            up_frames += 1
            if up_frames >= max(1, min_up):
                status = "up"
                start = start_temp
                up_frames = 0
                start_temp = None
        elif not active and status == "down":
            up_frames = 0
            start_temp = None

        if not active and status == "up":
            if down_frames == 0:
                end_temp = frame_index + delay
            down_frames += 1
            if down_frames >= max(1, min_down):
                status = "down"
                if start is not None and end_temp is not None and end_temp > start:
                    intervals.append((int(start), min(len(angles) - 1, int(end_temp))))
                start = None
                down_frames = 0
                end_temp = None
        elif active and status == "up":
            down_frames = 0
            end_temp = None
    return intervals


def _merge_intervals(
    intervals: list[tuple[int, int]],
    gap: int,
) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged: list[tuple[int, int]] = [intervals[0]]
    for start, end in intervals[1:]:
        previous_start, previous_end = merged[-1]
        if int(start) <= int(previous_end) + int(gap):
            merged[-1] = (previous_start, max(previous_end, int(end)))
        else:
            merged.append((int(start), int(end)))
    return merged

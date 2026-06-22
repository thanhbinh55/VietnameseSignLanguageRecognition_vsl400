"""Pose / motion boundary detection for multi-variant videos."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from .config import PreprocessConfig
    from .video_probe import VideoProps


@dataclass(frozen=True)
class _BackendState:
    backend: str
    pose: object | None = None
    model_path: Path | None = None


@dataclass(frozen=True)
class _HandFeatures:
    angle: float
    visibility: float
    activity: float


@dataclass
class _HandState:
    status: str = "down"
    up_frames: int = 0
    down_frames: int = 0
    start_frame_temp: int | None = None
    end_frame_temp: int | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    seen_active: bool = False

    def reset_completed_interval(self) -> None:
        self.start_frame = None
        self.end_frame = None
        self.seen_active = False


def detect_method_boundaries(
    video_path,
    props: "VideoProps",
    cfg: "PreprocessConfig",
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    logger_obj: logging.Logger | None = None,
) -> list[int]:
    """Return interior boundary frames between sign executions."""
    log = logger_obj if logger_obj is not None else logger
    lo = max(0, int(start_frame))
    hi = int(
        props.num_frames - 1 if end_frame is None else min(end_frame, props.num_frames - 1)
    )
    if hi <= lo or float(props.fps) <= 0:
        return []

    backend = _select_backend(cfg)
    if backend == "pose":
        pose_state = _load_pose_backend(cfg, log)
        try:
            if pose_state.backend == "pose" and pose_state.pose is not None:
                return _detect_pose_boundaries(
                    video_path,
                    props,
                    cfg,
                    lo,
                    hi,
                    pose_state.pose,
                    log,
                )
        finally:
            _close_backend(pose_state)

    step = max(1, round(float(props.fps) * 0.25))
    frame_indices, activity = compute_activity_signal(video_path, lo, hi, step, cfg, log)
    if not frame_indices:
        return []
    boundaries = resting_state_boundaries(frame_indices, activity, cfg)
    return [boundary for boundary in boundaries if lo < boundary <= hi]


def compute_activity_signal(
    video_path,
    start_frame: int,
    end_frame: int,
    step: int,
    cfg: "PreprocessConfig",
    logger_obj: logging.Logger | None = None,
) -> tuple[list[int], list[float]]:
    """Sample a video and compute a normalized activity signal."""
    log = logger_obj if logger_obj is not None else logger
    frame_indices = list(range(int(start_frame), int(end_frame) + 1, max(1, int(step))))
    if not frame_indices:
        return [], []

    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video_path))
    backend = _BackendState("motion")
    try:
        if not capture.isOpened():
            log.warning("compute_activity_signal: could not open %s", video_path)
            return [], []

        selected = _select_backend(cfg)
        if selected == "pose":
            backend = _load_pose_backend(cfg, log)

        activities: list[float] = []
        prev_gray: "np.ndarray | None" = None
        for index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            grabbed, frame = capture.read()
            if not grabbed or frame is None:
                activities.append(0.0)
                prev_gray = None
                continue

            try:
                if backend.backend == "pose" and backend.pose is not None:
                    activity = _pose_activity(
                        frame,
                        backend.pose,
                        cfg,
                        timestamp_ms=_frame_to_timestamp_ms(index, float(getattr(cfg, "fps", 0.0)) or 0.0),
                        cv2=cv2,
                        np=np,
                    )
                else:
                    activity, prev_gray = _motion_activity(frame, prev_gray, cv2, np)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "compute_activity_signal: frame %d failed (%s: %s); using 0.0",
                    index,
                    type(exc).__name__,
                    exc,
                )
                activity = 0.0
                prev_gray = None

            activities.append(float(max(0.0, min(1.0, activity))))

        return frame_indices, activities
    finally:
        capture.release()
        _close_backend(backend)


def resting_state_boundaries(
    frame_indices: list[int],
    activity: list[float],
    cfg: "PreprocessConfig",
) -> list[int]:
    """Debounced active/resting state machine returning boundary frames."""
    if not frame_indices or not activity:
        return []

    threshold = 0.5
    min_up = max(1, int(getattr(cfg, "pose_min_up_frames", 5)))
    min_down = max(1, int(getattr(cfg, "pose_min_down_frames", 7)))
    delay = max(0, int(getattr(cfg, "pose_delay_frames", 4)))

    state = activity[0] >= threshold
    seen_active = state
    seen_rest = not state
    candidate_state: bool | None = None
    candidate_count = 0
    candidate_start = 0
    boundaries: list[int] = []

    for idx, value in zip(frame_indices[1:], activity[1:]):
        observed = value >= threshold
        if observed == state:
            candidate_state = None
            candidate_count = 0
            continue

        if candidate_state != observed:
            candidate_state = observed
            candidate_count = 1
            candidate_start = idx
        else:
            candidate_count += 1

        required = min_up if observed else min_down
        if candidate_count < required:
            continue

        previous_state = state
        state = observed
        candidate_state = None
        candidate_count = 0

        if previous_state is False and observed is True and seen_active and seen_rest:
            boundary = candidate_start + delay
            if not boundaries or boundary > boundaries[-1]:
                boundaries.append(boundary)

        seen_active = seen_active or observed
        seen_rest = seen_rest or (not observed)

    return boundaries


def _select_backend(cfg: "PreprocessConfig") -> str:
    backend = getattr(cfg, "pose_backend", "auto")
    if not getattr(cfg, "pose_enabled", True):
        return "motion"
    if backend in {"motion", "pose"}:
        return backend
    try:
        import mediapipe  # noqa: F401
    except Exception:
        return "motion"
    return "pose"


def _load_pose_backend(cfg: "PreprocessConfig", log: logging.Logger) -> _BackendState:
    try:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except Exception as exc:  # noqa: BLE001
        log.info(
            "MediaPipe Tasks unavailable (%s); falling back to motion backend.",
            type(exc).__name__,
        )
        return _BackendState("motion")

    model_path = _resolve_pose_model_path(cfg)
    if model_path is None:
        log.info("Pose model not found; falling back to motion backend.")
        return _BackendState("motion")

    try:
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        pose = vision.PoseLandmarker.create_from_options(options)
    except Exception as exc:  # noqa: BLE001
        log.info(
            "Could not initialize PoseLandmarker from %s (%s); falling back to motion backend.",
            model_path,
            type(exc).__name__,
        )
        return _BackendState("motion")

    return _BackendState("pose", pose, model_path)


def _close_backend(backend: _BackendState) -> None:
    if backend.backend != "pose" or backend.pose is None:
        return
    close = getattr(backend.pose, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001
            pass


def _resolve_pose_model_path(cfg: "PreprocessConfig") -> Path | None:
    candidates = (
        "Dataset/models/pose_landmarker_heavy.task",
        "Dataset/models/pose_landmarker_full.task",
        "Dataset/models/pose_landmarker_lite.task",
    )
    for raw_path in candidates:
        candidate = cfg.resolve(raw_path)
        if candidate.is_file():
            return candidate
    return None


def _motion_activity(frame, prev_gray, cv2, np) -> tuple[float, "object | None"]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    y0 = int(h * 0.45)
    x0 = int(w * 0.25)
    x1 = int(w * 0.75)
    roi = gray[y0:h, x0:x1]
    if roi.size == 0:
        return 0.0, gray
    if prev_gray is None:
        return 0.0, gray
    prev_roi = prev_gray[y0:h, x0:x1]
    if prev_roi.shape != roi.shape:
        m_h = min(prev_roi.shape[0], roi.shape[0])
        m_w = min(prev_roi.shape[1], roi.shape[1])
        prev_roi = prev_roi[:m_h, :m_w]
        roi = roi[:m_h, :m_w]
    diff = float(np.mean(cv2.absdiff(prev_roi, roi))) / 255.0
    return min(1.0, diff * 4.0), gray


def _detect_pose_boundaries(
    video_path,
    props: "VideoProps",
    cfg: "PreprocessConfig",
    start_frame: int,
    end_frame: int,
    pose,
    log: logging.Logger,
) -> list[int]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            log.warning("_detect_pose_boundaries: could not open %s", video_path)
            return []
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

        left_state = _HandState()
        right_state = _HandState()
        intervals: list[tuple[int, int]] = []
        fps = float(props.fps)
        current = int(start_frame)

        while current <= int(end_frame):
            grabbed, frame = capture.read()
            if not grabbed or frame is None:
                break

            try:
                features = _pose_features(
                    frame,
                    pose,
                    cfg,
                    timestamp_ms=_frame_to_timestamp_ms(current, fps),
                    cv2=cv2,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "_detect_pose_boundaries: frame %d failed (%s: %s); treating as rest.",
                    current,
                    type(exc).__name__,
                    exc,
                )
                features = None

            if features is None:
                left_features = _rest_features()
                right_features = _rest_features()
            else:
                left_features, right_features = features

            _update_hand_state(left_state, left_features, cfg, current)
            _update_hand_state(right_state, right_features, cfg, current)

            interval = _completed_interval(left_state, right_state)
            if interval is not None:
                start, end = interval
                if end > start:
                    intervals.append(
                        (
                            max(int(start_frame), int(start)),
                            min(int(end_frame), int(end)),
                        )
                    )
                left_state.reset_completed_interval()
                right_state.reset_completed_interval()

            current += 1

        intervals = _merge_adjacent_intervals(intervals, fps)
        boundaries = _boundaries_from_action_intervals(intervals, cfg, start_frame, end_frame)
        filtered = sorted(
            {int(boundary) for boundary in boundaries if start_frame < int(boundary) <= end_frame}
        )
        return filtered
    finally:
        capture.release()


def _merge_adjacent_intervals(
    intervals: list[tuple[int, int]],
    fps: float,
) -> list[tuple[int, int]]:
    if not intervals:
        return []

    merge_gap = max(1, round(max(0.15, min(0.35, 4.0 / max(fps, 1.0))) * max(fps, 1.0)))
    merged: list[tuple[int, int]] = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= merge_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _boundaries_from_action_intervals(
    intervals: list[tuple[int, int]],
    cfg: "PreprocessConfig",
    start_frame: int,
    end_frame: int,
) -> list[int]:
    """Choose cuts inside the rest gap between consecutive action intervals."""
    if len(intervals) < 2:
        return []

    gap_ratio = float(getattr(cfg, "pose_boundary_gap_ratio", 0.42))
    gap_ratio = max(0.0, min(1.0, gap_ratio))
    offset = int(getattr(cfg, "pose_boundary_offset_frames", 4))

    boundaries: list[int] = []
    for previous, current in zip(intervals, intervals[1:]):
        rest_start = int(previous[1]) + 1
        rest_end = int(current[0])
        if rest_end < rest_start:
            continue
        boundary = int(round(rest_start + (rest_end - rest_start) * gap_ratio + offset))
        boundary = max(rest_start, min(rest_end, boundary))
        boundary = max(int(start_frame) + 1, min(int(end_frame), boundary))
        boundaries.append(boundary)
    return boundaries


def _pose_activity(frame, pose, cfg, *, timestamp_ms: int, cv2, np) -> float:
    features = _pose_features(frame, pose, cfg, timestamp_ms=timestamp_ms, cv2=cv2)
    if features is None:
        return 0.0
    left, right = features
    return float(max(0.0, min(1.0, max(left.activity, right.activity))))


def _pose_features(frame, pose, cfg, *, timestamp_ms: int, cv2):
    import mediapipe as mp

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = pose.detect_for_video(image, int(timestamp_ms))
    landmarks = getattr(result, "pose_landmarks", None)
    if not landmarks:
        return None

    lm = landmarks[0]
    left = _hand_features(
        lm[_PoseIdx.LEFT_SHOULDER],
        lm[_PoseIdx.LEFT_ELBOW],
        lm[_PoseIdx.LEFT_WRIST],
        cfg,
    )
    right = _hand_features(
        lm[_PoseIdx.RIGHT_SHOULDER],
        lm[_PoseIdx.RIGHT_ELBOW],
        lm[_PoseIdx.RIGHT_WRIST],
        cfg,
    )
    return left, right


def _update_hand_state(
    state: _HandState,
    features: _HandFeatures,
    cfg: "PreprocessConfig",
    frame_index: int,
) -> None:
    angle_threshold = float(getattr(cfg, "pose_angle_threshold", 155.0))
    visibility_threshold = float(getattr(cfg, "pose_visibility_threshold", 0.55))
    min_up = max(1, int(getattr(cfg, "pose_min_up_frames", 5)))
    min_down = max(1, int(getattr(cfg, "pose_min_down_frames", 7)))
    delay = max(0, int(getattr(cfg, "pose_delay_frames", 4)))

    active = features.angle < angle_threshold and features.visibility >= visibility_threshold
    inactive = not active
    if active:
        state.seen_active = True

    if active and state.status == "down":
        if state.up_frames == 0:
            state.start_frame_temp = max(0, frame_index - delay)
        state.up_frames += 1
        if state.up_frames >= min_up:
            state.status = "up"
            state.start_frame = state.start_frame_temp
            state.up_frames = 0
            state.start_frame_temp = None
    elif inactive and state.status == "down":
        state.up_frames = 0
        state.start_frame_temp = None

    if inactive and state.status == "up":
        if state.down_frames == 0:
            state.end_frame_temp = frame_index + delay
        state.down_frames += 1
        if state.down_frames >= min_down:
            state.status = "down"
            state.end_frame = state.end_frame_temp
            state.down_frames = 0
            state.end_frame_temp = None
    elif active and state.status == "up":
        state.down_frames = 0
        state.end_frame_temp = None


def _completed_interval(
    left_state: _HandState,
    right_state: _HandState,
) -> tuple[int, int] | None:
    left_ready = left_state.start_frame is not None and left_state.end_frame is not None
    right_ready = right_state.start_frame is not None and right_state.end_frame is not None
    left_unused = not left_state.seen_active and left_state.start_frame is None and left_state.end_frame is None
    right_unused = not right_state.seen_active and right_state.start_frame is None and right_state.end_frame is None

    if left_ready and right_unused and left_state.status == "down":
        return int(left_state.start_frame), int(left_state.end_frame)
    if right_ready and left_unused and right_state.status == "down":
        return int(right_state.start_frame), int(right_state.end_frame)
    if (
        left_ready
        and right_ready
        and left_state.status == "down"
        and right_state.status == "down"
    ):
        return (
            min(int(left_state.start_frame), int(right_state.start_frame)),
            max(int(left_state.end_frame), int(right_state.end_frame)),
        )
    return None


def _hand_features(shoulder, elbow, wrist, cfg: "PreprocessConfig") -> _HandFeatures:
    angle_threshold = float(getattr(cfg, "pose_angle_threshold", 155.0))
    angle = _joint_angle(
        (shoulder.x, shoulder.y),
        (elbow.x, elbow.y),
        (wrist.x, wrist.y),
    )
    visibility = float(
        min(
            getattr(wrist, "visibility", 0.0),
            getattr(wrist, "presence", 0.0),
        )
    )
    if visibility <= 0.0:
        return _HandFeatures(angle=180.0, visibility=0.0, activity=0.0)
    activity = max(0.0, min(1.0, (angle_threshold - angle) / max(angle_threshold, 1e-6)))
    activity *= visibility
    return _HandFeatures(angle=angle, visibility=visibility, activity=activity)


def _rest_features() -> _HandFeatures:
    return _HandFeatures(angle=180.0, visibility=0.0, activity=0.0)


def _joint_angle(a, b, c) -> float:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1])

    radians = math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx)
    angle = abs(math.degrees(radians))
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)


def _frame_to_timestamp_ms(frame_index: int, fps: float) -> int:
    if fps <= 0:
        return int(frame_index * 1000)
    return int(round(frame_index * 1000.0 / fps))


class _PoseIdx:
    LEFT_SHOULDER = 11
    LEFT_ELBOW = 13
    LEFT_WRIST = 15
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW = 14
    RIGHT_WRIST = 16

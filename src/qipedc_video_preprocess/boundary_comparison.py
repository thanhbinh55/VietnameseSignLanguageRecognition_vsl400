"""Boundary-method comparison against ``Dataset/true_label/``.

This module keeps the scoring logic separate from the CLI wrapper so the parsing,
ground-truth construction, and summary statistics can be tested without running
the full video pipeline.
"""

from __future__ import annotations

import logging
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from .boundary_calibration import predict_calibrated_boundaries
from .config import PreprocessConfig
from .discovery import VideoEntry
from .number_detector import EasyOcrNumberDetector
from .pose_boundary import detect_method_boundaries
from .segmenter import SegmentationResult, segment_video
from .video_probe import VideoProps, probe

logger = logging.getLogger(__name__)

_TRUE_LABEL_RE = re.compile(r"^(?P<base>.+?)(?:_c(?P<variant>\d+))?$")
_SOURCE_DIR_CANDIDATES = (
    "Dataset/processed_videos/processed_videos/resize_720p",
    "Dataset/processed_videos/resize_720p",
    "Dataset/raw_videos",
)


@dataclass(frozen=True)
class TrueLabelClip:
    base_id: str
    video_id: str
    path: Path
    variant_index: int | None
    is_side: bool
    num_frames: int
    fps: float


@dataclass(frozen=True)
class BoundaryCase:
    base_id: str
    source_path: Path
    source_props: VideoProps
    front_clips: tuple[TrueLabelClip, ...]
    side_clips: tuple[TrueLabelClip, ...]
    gt_boundaries: tuple[int, ...]

    @property
    def is_control(self) -> bool:
        return len(self.front_clips) <= 1


@dataclass(frozen=True)
class BoundaryPrediction:
    method: str
    predicted_boundaries: tuple[int, ...]
    status: str
    note: str = ""
    inferred: bool = False


@dataclass(frozen=True)
class BoundaryMatch:
    video_id: str
    method: str
    gt_boundaries: tuple[int, ...]
    predicted_boundaries: tuple[int, ...]
    matched_errors_frames: tuple[int, ...]
    matched_errors_seconds: tuple[float, ...]
    missed_boundaries: int
    extra_boundaries: int
    false_split: bool
    fps: float
    note: str = ""
    status: str = "ok"

    @property
    def matched_count(self) -> int:
        return len(self.matched_errors_frames)


@dataclass
class MethodSummary:
    method: str
    cases: int = 0
    control_cases: int = 0
    total_gt_boundaries: int = 0
    total_pred_boundaries: int = 0
    matched_boundaries: int = 0
    missed_boundaries: int = 0
    extra_boundaries: int = 0
    false_splits: int = 0
    abs_errors_frames: list[int] | None = None
    abs_errors_seconds: list[float] | None = None
    within_tolerance: int = 0
    skipped: int = 0
    unavailable: int = 0

    def __post_init__(self) -> None:
        if self.abs_errors_frames is None:
            self.abs_errors_frames = []
        if self.abs_errors_seconds is None:
            self.abs_errors_seconds = []

    @property
    def scored_cases(self) -> int:
        return self.cases - self.skipped


def parse_true_label_name(filename: str | Path) -> tuple[str, int | None, bool]:
    """Return ``(base_id, variant_index, is_side)`` for a true-label filename."""
    stem = Path(filename).stem
    is_side = stem.endswith("_side")
    core = stem[:-5] if is_side else stem
    match = _TRUE_LABEL_RE.match(core)
    if not match:
        raise ValueError(f"Cannot parse true-label filename: {filename!r}")
    base_id = match.group("base")
    variant_raw = match.group("variant")
    variant_index = int(variant_raw) if variant_raw is not None else None
    return base_id, variant_index, is_side


def resolve_source_video_path(base_id: str, cfg: PreprocessConfig) -> Path | None:
    """Locate the source video for *base_id* across the known video roots."""
    roots = [cfg.resolve(path) for path in _SOURCE_DIR_CANDIDATES]
    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        candidate = root / f"{base_id}.mp4"
        if candidate.is_file():
            return candidate
        if root.exists():
            try:
                found = next(root.rglob(f"{base_id}.mp4"))
            except StopIteration:
                found = None
            if found is not None:
                return found
    return None


def load_boundary_cases(
    true_label_dir: Path,
    cfg: PreprocessConfig,
    *,
    logger_obj: logging.Logger | None = None,
) -> list[BoundaryCase]:
    """Build boundary-comparison cases from ``Dataset/true_label/``."""
    log = logger_obj if logger_obj is not None else logger
    if not true_label_dir.is_dir():
        raise FileNotFoundError(f"Missing true-label directory: {true_label_dir}")

    grouped: dict[str, list[TrueLabelClip]] = defaultdict(list)
    skipped = 0
    for clip_path in sorted(true_label_dir.glob("*.mp4")):
        try:
            base_id, variant_index, is_side = parse_true_label_name(clip_path)
        except ValueError as exc:
            log.warning("Skipping %s (%s)", clip_path.name, exc)
            skipped += 1
            continue

        clip_props = probe(clip_path)
        if clip_props is None:
            log.warning("Skipping unreadable true-label clip %s", clip_path)
            skipped += 1
            continue

        grouped[base_id].append(
            TrueLabelClip(
                base_id=base_id,
                video_id=clip_path.stem,
                path=clip_path,
                variant_index=variant_index,
                is_side=is_side,
                num_frames=int(clip_props.num_frames),
                fps=float(clip_props.fps),
            )
        )

    cases: list[BoundaryCase] = []
    for base_id in sorted(grouped):
        source_path = resolve_source_video_path(base_id, cfg)
        if source_path is None:
            log.warning("Skipping %s: source video not found.", base_id)
            continue
        source_props = probe(source_path)
        if source_props is None:
            log.warning("Skipping %s: source video unreadable (%s).", base_id, source_path)
            continue

        clips = grouped[base_id]
        front_clips = _sorted_front_clips(clips)
        if not front_clips:
            log.warning("Skipping %s: no front clips found in true_label.", base_id)
            continue

        side_clips = tuple(sorted((clip for clip in clips if clip.is_side), key=lambda c: c.video_id))
        gt_boundaries = _ground_truth_boundaries(front_clips)
        cases.append(
            BoundaryCase(
                base_id=base_id,
                source_path=source_path,
                source_props=source_props,
                front_clips=front_clips,
                side_clips=side_clips,
                gt_boundaries=gt_boundaries,
            )
        )

    log.info(
        "Loaded %d boundary cases from %s (%d skipped clips).",
        len(cases),
        true_label_dir,
        skipped,
    )
    return cases


def compare_boundary_methods(
    cfg: PreprocessConfig,
    cases: list[BoundaryCase],
    *,
    methods: tuple[str, ...] = ("ocr", "pose", "ensemble", "calibrated"),
    tolerance_seconds: float | None = None,
    logger_obj: logging.Logger | None = None,
) -> tuple[list[BoundaryMatch], dict[str, MethodSummary]]:
    """Score the requested boundary methods against the supplied cases."""
    log = logger_obj if logger_obj is not None else logger
    tolerance = float(
        cfg.ensemble_tolerance_seconds if tolerance_seconds is None else tolerance_seconds
    )
    matches: list[BoundaryMatch] = []
    summaries = {method: MethodSummary(method=method) for method in methods}

    ocr_detector = None
    ocr_available, ocr_reason = _ocr_stack_available()
    if ocr_available and any(method in {"ocr", "ensemble"} for method in methods):
        try:
            ocr_detector = EasyOcrNumberDetector(cfg=cfg, log=log)
        except Exception as exc:  # noqa: BLE001
            ocr_available = False
            ocr_reason = f"{type(exc).__name__}: {exc}"

    for case in cases:
        for method in methods:
            summary = summaries[method]
            summary.cases += 1
            if case.is_control:
                summary.control_cases += 1

            if method in {"ocr", "ensemble"} and not ocr_available:
                summary.skipped += 1
                summary.unavailable += 1
                matches.append(
                    BoundaryMatch(
                        video_id=case.base_id,
                        method=method,
                        gt_boundaries=case.gt_boundaries,
                        predicted_boundaries=(),
                        matched_errors_frames=(),
                        matched_errors_seconds=(),
                        missed_boundaries=len(case.gt_boundaries),
                        extra_boundaries=0,
                        false_split=False,
                        fps=float(case.source_props.fps),
                        note=ocr_reason or "OCR stack unavailable",
                        status="unavailable",
                    )
                )
                continue

            prediction = predict_case_boundaries(
                case,
                method=method,
                cfg=cfg,
                ocr_detector=ocr_detector,
                logger_obj=log,
            )
            match = score_prediction(case, method, prediction, tolerance)
            matches.append(match)
            _accumulate_summary(summary, match, tolerance)

    return matches, summaries


def predict_case_boundaries(
    case: BoundaryCase,
    *,
    method: str,
    cfg: PreprocessConfig,
    ocr_detector: EasyOcrNumberDetector | None,
    logger_obj: logging.Logger | None = None,
) -> BoundaryPrediction:
    """Predict boundary frames for one case with the selected method."""
    log = logger_obj if logger_obj is not None else logger
    method = method.lower().strip()
    if method == "pose":
        boundaries = detect_method_boundaries(
            case.source_path,
            case.source_props,
            cfg,
            start_frame=0,
            end_frame=max(0, int(case.source_props.num_frames) - 1),
            logger_obj=log,
        )
        return BoundaryPrediction(
            method=method,
            predicted_boundaries=tuple(boundaries),
            status="ok",
            inferred=False,
        )

    if method == "calibrated":
        calibrated = predict_calibrated_boundaries(
            case.base_id,
            case.source_path,
            case.source_props,
            cfg,
            expected_count=len(case.gt_boundaries),
            logger_obj=log,
        )
        return BoundaryPrediction(
            method=method,
            predicted_boundaries=calibrated.boundaries,
            status=calibrated.status,
            note=calibrated.note,
            inferred=not calibrated.label_fitted,
        )

    if method not in {"ocr", "ensemble"}:
        raise ValueError(f"Unsupported boundary method: {method!r}")

    if ocr_detector is None:
        return BoundaryPrediction(
            method=method,
            predicted_boundaries=(),
            status="unavailable",
            note="OCR detector not initialized",
        )

    entry = VideoEntry(
        video_id=case.base_id,
        path=case.source_path,
        source_dir=str(case.source_path.parent),
    )
    method_cfg = replace(cfg, boundary_method=method)
    result = segment_video(
        entry,
        case.source_props,
        ocr_detector,
        method_cfg,
        logger=log,
    )
    boundaries = tuple(span.start_frame for span in result.spans[1:]) if result.kind == "multi" else ()
    return BoundaryPrediction(
        method=method,
        predicted_boundaries=boundaries,
        status="ok",
        inferred=bool(getattr(result, "inferred", False)),
    )


def score_prediction(
    case: BoundaryCase,
    method: str,
    prediction: BoundaryPrediction,
    tolerance_seconds: float,
) -> BoundaryMatch:
    """Compare one prediction against a ground-truth boundary list."""
    gt = case.gt_boundaries
    pred = prediction.predicted_boundaries
    matched = min(len(gt), len(pred))
    fps = float(case.source_props.fps)
    errors_frames = tuple(abs(int(pred[i]) - int(gt[i])) for i in range(matched))
    errors_seconds = tuple((err / fps) if fps > 0 else 0.0 for err in errors_frames)
    missed = max(0, len(gt) - len(pred))
    extra = max(0, len(pred) - len(gt))
    false_split = case.is_control and bool(pred)
    status = prediction.status
    return BoundaryMatch(
        video_id=case.base_id,
        method=method,
        gt_boundaries=gt,
        predicted_boundaries=pred,
        matched_errors_frames=errors_frames,
        matched_errors_seconds=errors_seconds,
        missed_boundaries=missed,
        extra_boundaries=extra,
        false_split=false_split,
        fps=fps,
        note=prediction.note,
        status=status,
    )


def summarize_method(
    method: str,
    matches: list[BoundaryMatch],
    *,
    tolerance_seconds: float,
) -> MethodSummary:
    """Aggregate a list of matches into a compact summary."""
    summary = MethodSummary(method=method)
    summary.cases = len(matches)
    summary.control_cases = len(matches)
    for match in matches:
        _accumulate_summary(summary, match, tolerance_seconds)
    return summary


def format_case_table(matches: list[BoundaryMatch], *, limit: int | None = None) -> str:
    """Return a human-readable per-video table."""
    rows = matches if limit is None else matches[:limit]
    header = (
        f"{'method':<9} {'video_id':<10} {'gt':<14} {'pred':<14} "
        f"{'err(fr)':<14} {'miss':<5} {'extra':<5} note"
    )
    lines = [header, "-" * len(header)]
    for match in rows:
        errors = ",".join(str(value) for value in match.matched_errors_frames) or "-"
        lines.append(
            f"{match.method:<9} {match.video_id:<10} "
            f"{_fmt_list(match.gt_boundaries):<14} {_fmt_list(match.predicted_boundaries):<14} "
            f"{errors:<14} {match.missed_boundaries:<5} {match.extra_boundaries:<5} "
            f"{match.note or match.status}"
        )
    if limit is not None and len(matches) > limit:
        lines.append(f"... ({len(matches) - limit} more rows)")
    return "\n".join(lines)


def format_summary_table(summaries: dict[str, MethodSummary], tolerance_seconds: float) -> str:
    """Return a compact aggregate metrics table."""
    header = (
        f"{'method':<9} {'cases':<5} {'scored':<6} {'gt':<5} {'found':<5} "
        f"{'miss':<5} {'extra':<5} {'false':<5} {'det%':<7} {'MAE(fr)':<8} {'med(fr)':<8} "
        f"{'MAE(s)':<8} {'med(s)':<8} {'<=tol':<7}"
    )
    lines = [header, "-" * len(header)]
    for method in sorted(summaries):
        summary = summaries[method]
        mae_frames = _mean(summary.abs_errors_frames)
        med_frames = _median(summary.abs_errors_frames)
        mae_seconds = _mean(summary.abs_errors_seconds)
        med_seconds = _median(summary.abs_errors_seconds)
        detection_rate = (
            f"{(summary.matched_boundaries / summary.total_gt_boundaries * 100.0):.1f}%"
            if summary.total_gt_boundaries
            else "-"
        )
        within_tol = (
            f"{(summary.within_tolerance / len(summary.abs_errors_seconds) * 100.0):.1f}%"
            if summary.abs_errors_seconds
            else "-"
        )
        lines.append(
            f"{method:<9} {summary.cases:<5} {summary.scored_cases:<6} "
            f"{summary.total_gt_boundaries:<5} {summary.matched_boundaries:<5} "
            f"{summary.missed_boundaries:<5} {summary.extra_boundaries:<5} "
            f"{summary.false_splits:<5} {detection_rate:<7} {_fmt_num(mae_frames):<8} {_fmt_num(med_frames):<8} "
            f"{_fmt_num(mae_seconds):<8} {_fmt_num(med_seconds):<8} {within_tol:<7}"
        )
    lines.append(f"Tolerance: ±{tolerance_seconds:.3f}s")
    return "\n".join(lines)


def recommend_method(summaries: dict[str, MethodSummary]) -> str | None:
    """Return the provisional best method based on aggregate error statistics."""
    ranked: list[tuple[tuple[float, float, float, float, float], str]] = []
    for method, summary in summaries.items():
        if summary.scored_cases <= 0 or not summary.abs_errors_seconds:
            continue
        mean_err = _mean(summary.abs_errors_seconds)
        median_err = _median(summary.abs_errors_seconds)
        detection_rate = summary.matched_boundaries / summary.total_gt_boundaries if summary.total_gt_boundaries else 0.0
        false_split_rate = summary.false_splits / summary.control_cases if summary.control_cases else 0.0
        ranked.append(
            (
                (
                    mean_err,
                    median_err,
                    -detection_rate,
                    false_split_rate,
                    summary.missed_boundaries + summary.extra_boundaries,
                ),
                method,
            )
        )
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][1]


def _sorted_front_clips(clips: list[TrueLabelClip]) -> tuple[TrueLabelClip, ...]:
    fronts = [clip for clip in clips if not clip.is_side]
    if not fronts:
        return ()
    return tuple(
        sorted(
            fronts,
            key=lambda clip: (
                clip.variant_index if clip.variant_index is not None else 1_000_000,
                clip.video_id,
            ),
        )
    )


def _ground_truth_boundaries(front_clips: tuple[TrueLabelClip, ...]) -> tuple[int, ...]:
    boundaries: list[int] = []
    running = 0
    for clip in front_clips[:-1]:
        running += int(clip.num_frames)
        boundaries.append(running)
    return tuple(boundaries)


def _ocr_stack_available() -> tuple[bool, str | None]:
    try:
        import easyocr  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def _accumulate_summary(summary: MethodSummary, match: BoundaryMatch, tolerance_seconds: float) -> None:
    summary.total_gt_boundaries += len(match.gt_boundaries)
    summary.total_pred_boundaries += len(match.predicted_boundaries)
    summary.matched_boundaries += match.matched_count
    summary.missed_boundaries += match.missed_boundaries
    summary.extra_boundaries += match.extra_boundaries
    summary.false_splits += 1 if match.false_split else 0
    summary.abs_errors_frames.extend(match.matched_errors_frames)
    summary.abs_errors_seconds.extend(match.matched_errors_seconds)
    summary.within_tolerance += sum(1 for value in match.matched_errors_seconds if value <= tolerance_seconds)


def _fmt_list(values: tuple[int, ...]) -> str:
    return "[" + ",".join(str(v) for v in values) + "]" if values else "[]"


def _fmt_num(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _mean(values: list[float] | list[int]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _median(values: list[float] | list[int]) -> float | None:
    return float(statistics.median(values)) if values else None

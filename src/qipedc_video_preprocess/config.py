"""Configuration for the QIPEDC preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REQUIRED_DRIVE = "D:"


@dataclass(frozen=True)
class PreprocessConfig:
    project_root: Path

    required_drive: str | None = None

    video_search_dirs: tuple[str, ...] = (
        "Dataset/processed_videos/resize_720p",
        "Dataset/raw_videos",
    )

    labels_glob: str = "Dataset/labels/*.xlsx"

    split_output_dir: str = "Dataset/processed_videos/split_variants"
    new_labels_path: str = "Dataset/processed_videos/labels_split.xlsx"
    log_dir: str = "Dataset/logs"
    manual_review_dir: str = "Dataset/processed_videos/manual"
    inferred_review_dir: str = "Dataset/processed_videos/split_variant_predicted"
    completion_dir: str = "Dataset/processed_videos/.split_completed"
    view_inventory_path: str | None = None

    roi_top_left: tuple[float, float, float, float] = (0.19, 0.05, 0.25, 0.22)
    ocr_confidence_threshold: float = 0.5
    ocr_models_dir: str = "Dataset/models/easyocr"
    ocr_gpu: bool = False

    # Keep the conservative single-process behavior by default. Batch runners
    # can opt in after verifying that their machine has enough CPU/RAM.
    parallel_workers: int = 1
    resume_existing: bool = False

    sample_interval_seconds: float = 1.0
    boundary_confirm_frames: int = 2
    min_variant_seconds: float = 0.6

    multiview_enabled: bool = True
    multiview_diff_threshold: float = 0.01
    multiview_min_gap_seconds: float = 0.8

    boundary_method: str = "ensemble"
    boundary_calibration_path: str = "Dataset/logs/pose_boundary_calibration.json"
    pose_enabled: bool = True
    pose_backend: str = "auto"
    pose_angle_threshold: float = 155.0
    pose_visibility_threshold: float = 0.55
    pose_min_up_frames: int = 5
    pose_min_down_frames: int = 7
    pose_delay_frames: int = 4
    pose_boundary_gap_ratio: float = 0.42
    pose_boundary_offset_frames: int = 4
    ensemble_tolerance_seconds: float = 0.6
    view_ensemble_tolerance_seconds: float = 0.3
    view_predicted_tolerance_seconds: float = 0.6

    safety_margin_frames: int = 3

    def resolve(self, relative_path: str) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute():
            return candidate.resolve()
        return (Path(self.project_root) / candidate).resolve()

    @property
    def split_output_path(self) -> Path:
        return self.resolve(self.split_output_dir)

    @property
    def new_labels_full_path(self) -> Path:
        return self.resolve(self.new_labels_path)

    @property
    def log_path(self) -> Path:
        return self.resolve(self.log_dir)

    @property
    def manual_review_path(self) -> Path:
        return self.resolve(self.manual_review_dir)

    @property
    def inferred_review_path(self) -> Path:
        return self.resolve(self.inferred_review_dir)

    @property
    def completion_path(self) -> Path:
        return self.resolve(self.completion_dir)

    @property
    def view_inventory_full_path(self) -> Path | None:
        if self.view_inventory_path is None:
            return None
        return self.resolve(self.view_inventory_path)

    @property
    def ocr_models_path(self) -> Path:
        return self.resolve(self.ocr_models_dir)

    def video_search_paths(self) -> tuple[Path, ...]:
        return tuple(self.resolve(d) for d in self.video_search_dirs)

    @property
    def boundary_method_value(self) -> str:
        return self.boundary_method

    def validate(self) -> list[str]:
        errors: list[str] = []

        try:
            root = Path(self.project_root).resolve()
        except (OSError, ValueError) as exc:
            errors.append(
                f"project_root could not be resolved: {self.project_root!r} ({exc})"
            )
            root = None

        if (
            root is not None
            and self.required_drive is not None
            and not _is_on_drive(root, self.required_drive)
        ):
            errors.append(
                "project_root must be on drive "
                f"{self.required_drive!r} but is {root!r}."
            )

        for field_name, raw_value in (
            ("split_output_dir", self.split_output_dir),
            ("new_labels_path", self.new_labels_path),
            ("log_dir", self.log_dir),
            ("manual_review_dir", self.manual_review_dir),
            ("inferred_review_dir", self.inferred_review_dir),
            ("completion_dir", self.completion_dir),
            ("boundary_calibration_path", self.boundary_calibration_path),
        ):
            errors.extend(self._validate_output_path(field_name, raw_value, root))

        if self.view_inventory_path is not None:
            try:
                inventory = self.resolve(self.view_inventory_path)
                if root is not None and not _is_within(inventory, root):
                    errors.append(
                        f"view_inventory_path={self.view_inventory_path!r} resolved "
                        f"to {inventory!r} outside {root!r}."
                    )
                elif not inventory.is_file():
                    errors.append(f"view_inventory_path does not exist: {inventory!r}.")
            except (OSError, ValueError) as exc:
                errors.append(
                    f"view_inventory_path={self.view_inventory_path!r} could not be "
                    f"resolved ({exc})."
                )

        if self.boundary_method not in {"ocr", "pose", "ensemble", "calibrated"}:
            errors.append(
                "boundary_method must be one of {'ocr', 'pose', 'ensemble', 'calibrated'}."
            )

        if self.pose_backend not in {"auto", "pose", "motion"}:
            errors.append(
                "pose_backend must be one of {'auto', 'pose', 'motion'}."
            )

        if not _is_real_number(self.pose_angle_threshold) or not (
            0.0 <= float(self.pose_angle_threshold) <= 180.0
        ):
            errors.append("pose_angle_threshold must be in [0.0, 180.0].")

        if not _is_real_number(self.pose_visibility_threshold) or not (
            0.0 <= float(self.pose_visibility_threshold) <= 1.0
        ):
            errors.append("pose_visibility_threshold must be in [0.0, 1.0].")

        for field_name, value in (
            ("pose_min_up_frames", self.pose_min_up_frames),
            ("pose_min_down_frames", self.pose_min_down_frames),
        ):
            if not _is_integer(value) or int(value) <= 0:
                errors.append(f"{field_name} must be a positive integer.")

        if not _is_integer(self.pose_delay_frames) or int(self.pose_delay_frames) < 0:
            errors.append("pose_delay_frames must be a non-negative integer.")

        if not _is_real_number(self.pose_boundary_gap_ratio) or not (
            0.0 <= float(self.pose_boundary_gap_ratio) <= 1.0
        ):
            errors.append("pose_boundary_gap_ratio must be in [0.0, 1.0].")

        if not _is_integer(self.pose_boundary_offset_frames) or not (
            -120 <= int(self.pose_boundary_offset_frames) <= 120
        ):
            errors.append("pose_boundary_offset_frames must be an integer in [-120, 120].")

        if not _is_real_number(self.ensemble_tolerance_seconds) or not (
            0.0 <= float(self.ensemble_tolerance_seconds) <= 5.0
        ):
            errors.append("ensemble_tolerance_seconds must be in [0.0, 5.0].")

        if not _is_real_number(self.view_ensemble_tolerance_seconds) or not (
            0.0 <= float(self.view_ensemble_tolerance_seconds) <= 5.0
        ):
            errors.append("view_ensemble_tolerance_seconds must be in [0.0, 5.0].")

        if (
            not _is_real_number(self.view_predicted_tolerance_seconds)
            or not _is_real_number(self.view_ensemble_tolerance_seconds)
            or not (
                float(self.view_ensemble_tolerance_seconds)
                <= float(self.view_predicted_tolerance_seconds)
                <= 5.0
            )
        ):
            errors.append(
                "view_predicted_tolerance_seconds must be between "
                "view_ensemble_tolerance_seconds and 5.0."
            )

        if not _is_real_number(self.ocr_confidence_threshold) or not (
            0.0 <= float(self.ocr_confidence_threshold) <= 1.0
        ):
            errors.append("ocr_confidence_threshold must be in [0.0, 1.0].")

        if not isinstance(self.ocr_gpu, bool):
            errors.append("ocr_gpu must be a boolean.")

        if not _is_integer(self.parallel_workers) or not (
            1 <= int(self.parallel_workers) <= 16
        ):
            errors.append("parallel_workers must be in [1, 16].")

        if not isinstance(self.resume_existing, bool):
            errors.append("resume_existing must be a boolean.")

        if not _is_real_number(self.sample_interval_seconds) or not (
            0.1 <= float(self.sample_interval_seconds) <= 5.0
        ):
            errors.append("sample_interval_seconds must be in [0.1, 5.0].")

        if not _is_integer(self.safety_margin_frames) or not (
            0 <= int(self.safety_margin_frames) <= 60
        ):
            errors.append("safety_margin_frames must be an integer in [0, 60].")

        errors.extend(self._validate_roi())
        return errors

    def _validate_output_path(
        self, field_name: str, raw_value: str, root: Path | None
    ) -> list[str]:
        errors: list[str] = []
        try:
            resolved = self.resolve(raw_value)
        except (OSError, ValueError) as exc:
            errors.append(f"{field_name}={raw_value!r} could not be resolved ({exc}).")
            return errors

        if self.required_drive is not None and not _is_on_drive(
            resolved, self.required_drive
        ):
            errors.append(
                f"{field_name}={raw_value!r} resolved to {resolved!r}, not drive "
                f"{self.required_drive!r}."
            )

        if root is not None and not _is_within(resolved, root):
            errors.append(
                f"{field_name}={raw_value!r} resolved to {resolved!r} outside {root!r}."
            )
        return errors

    def _validate_roi(self) -> list[str]:
        errors: list[str] = []
        roi = self.roi_top_left
        if not isinstance(roi, (tuple, list)) or len(roi) != 4:
            errors.append("roi_top_left must be a 4-tuple.")
            return errors

        x0, y0, x1, y1 = roi
        coords = (("x0", x0), ("y0", y0), ("x1", x1), ("y1", y1))
        for name, value in coords:
            if not _is_real_number(value) or not (0.0 <= float(value) <= 1.0):
                errors.append(f"roi_top_left.{name} must be in [0.0, 1.0].")

        if all(_is_real_number(v) for _, v in coords):
            if not float(x0) < float(x1):
                errors.append("roi_top_left requires x0 < x1.")
            if not float(y0) < float(y1):
                errors.append("roi_top_left requires y0 < y1.")

        return errors


def _is_on_drive(path: Path, drive: str) -> bool:
    return path.drive.upper() == str(drive).upper()


def _is_on_required_drive(path: Path) -> bool:
    return _is_on_drive(path, REQUIRED_DRIVE)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_real_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return value == value


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)

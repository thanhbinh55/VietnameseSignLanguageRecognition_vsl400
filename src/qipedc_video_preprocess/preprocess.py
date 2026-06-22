"""Bộ_Tiền_Xử_Lý — CLI orchestrator điều phối toàn bộ công đoạn (Req 1, 9).

Nối các bước theo đúng thiết kế:

``cfg.validate()`` → nếu có lỗi: ghi lỗi cấu hình & **dừng trước khi xử lý bất kỳ
video nào, không tạo file đầu ra** (Req 1.4) → ``discover_videos`` → nếu rỗng:
cảnh báo & kết thúc (Req 2.6) → mỗi video: ``probe`` + ``segment_video`` →
``plan_outputs`` (xung đột tên → manual_review, Req 6.5) → ``write_clip`` từng
clip → ``build_new_rows`` + ``write_new_labels`` → ghi ``RunReport``.

Tính idempotent (Req 9) đạt được nhờ tên file & STT là hàm xác định của nguồn,
và luôn ghi đè cùng tên thay vì thêm hậu tố.

Công đoạn **không** tạo/sửa/xóa file thuộc ``src/qipedc2vsl400/`` hay
``Dataset/final_dataset/`` (Req 1.5): nó chỉ ghi vào ``split_output_dir``,
``new_labels_path`` và ``log_dir`` đã được ``validate()`` ràng buộc nằm trong cây
dự án trên ``D:``.

CLI::

    python -m qipedc_video_preprocess.preprocess
        [--project-root D:/...] [--sample-interval 1.0] [--safety-margin 3]
        [--ocr-threshold 0.5] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from .config import PreprocessConfig
from .discovery import discover_videos
from .label_writer import build_new_rows, read_source_labels, write_new_labels
from .multiview_detector import (
    detect_hardcut,
    detect_hardcut_candidate,
    detect_hardcut_near,
)
from .run_logger import RunReport, build_run_report, get_run_logger
from .segmenter import VariantSpan, segment_video
from .splitter import OutputClip, plan_outputs, trimmed_spans, view_filename, write_clip


# Increment whenever boundary/output semantics change so resume never reuses
# clips produced by an older algorithm.
OUTPUT_MANIFEST_VERSION = 3


@dataclass(frozen=True)
class OutputWriteJob:
    """Pickle-safe work item for one method clip."""

    src_path: Path
    clip: OutputClip
    seg: object
    props: object
    cfg: PreprocessConfig


@dataclass(frozen=True)
class OutputWriteResult:
    """Serializable result returned from an output worker."""

    video_id: str
    written: int
    label_clips: tuple[OutputClip, ...]
    manual_review: bool
    messages: tuple[tuple[str, str], ...]


class _WorkerLog:
    """Collect worker log messages for deterministic replay in the parent."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    @staticmethod
    def _render(message, args) -> str:
        if not args:
            return str(message)
        try:
            return str(message) % args
        except Exception:  # noqa: BLE001 - logging must never break a job
            return " ".join([str(message), *(str(arg) for arg in args)])

    def _add(self, level: str, message, *args) -> None:
        self.messages.append((level, self._render(message, args)))

    def debug(self, message, *args, **_kwargs) -> None:
        self._add("debug", message, *args)

    def info(self, message, *args, **_kwargs) -> None:
        self._add("info", message, *args)

    def warning(self, message, *args, **_kwargs) -> None:
        self._add("warning", message, *args)

    def error(self, message, *args, **_kwargs) -> None:
        self._add("error", message, *args)

    def exception(self, message, *args, **_kwargs) -> None:
        self._add("error", message, *args)


def run_preprocess(
    cfg: PreprocessConfig,
    detector=None,
    *,
    dry_run: bool = False,
    logger=None,
) -> RunReport:
    """Chạy toàn bộ công đoạn tách video cho cấu hình *cfg* (Req 1, 9).

    Args:
        cfg: :class:`PreprocessConfig` đã cấu hình.
        detector: Bộ phát hiện số (:class:`NumberDetector`). Nếu ``None`` và không
            ``dry_run``, một :class:`EasyOcrNumberDetector` được tạo từ ``cfg``.
            Tham số này cho phép inject detector giả trong integration test.
        dry_run: Nếu ``True``, không ghi file clip/nhãn (chỉ duyệt + phân đoạn +
            kế toán) — hữu ích để kiểm tra phân loại mà không tốn I/O.
        logger: Logger của lần chạy. Nếu ``None``, tạo bằng :func:`get_run_logger`
            (tạo ``Dataset/logs/`` nếu thiếu — Req 8.5). Inject được để test.

    Returns:
        :class:`RunReport` tóm tắt lần chạy.
    """
    # --- Bước 1: validate cấu hình TRƯỚC khi tạo bất kỳ file đầu ra nào (Req 1.4) ---
    config_errors = cfg.validate()
    if config_errors:
        # Không tạo logger/file log ở đây để tuân thủ "không tạo file đầu ra" khi
        # cấu hình sai (Req 1.4); báo lỗi ra stderr.
        for err in config_errors:
            print(f"[config error] {err}", file=sys.stderr)
        # Trả report rỗng; không có video nào được xử lý.
        return RunReport()

    log = logger if logger is not None else get_run_logger(cfg)

    # --- Bước 2: duyệt video nguồn ---
    entries = discover_videos(cfg, log)
    if not entries:
        log.warning("Không có video .mp4 nào để xử lý — kết thúc, không tạo đầu ra.")
        return RunReport()

    # Detector: tạo EasyOCR mặc định nếu cần (chỉ khi không dry-run và chưa inject).
    if detector is None and not dry_run:
        detector = _make_default_detector(cfg, log)

    discovered_ids = {entry.video_id: entry for entry in entries}

    # --- Bước 3: phân đoạn từng video ---
    from . import video_probe

    seg_results = []
    props_by_id: dict[str, object] = {}
    skipped = 0
    for entry in entries:
        props = video_probe.probe(entry.path)
        if props is None:
            log.warning(
                "Bỏ qua %s — không probe được thuộc tính video (video_id=%s).",
                entry.path,
                entry.video_id,
            )
            skipped += 1
            continue
        props_by_id[entry.video_id] = props

        if dry_run and detector is None:
            # Không có detector trong dry-run thuần: coi như single (không phân đoạn).
            from .segmenter import SegmentationResult

            result = SegmentationResult(
                video_id=entry.video_id,
                kind="single",
                variant_count=1,
                spans=(),
                observed_numbers=(),
            )
        else:
            result = segment_video(entry, props, detector, cfg, log)
        seg_results.append(result)

    # --- Bước 4: lên kế hoạch đầu ra + phát hiện xung đột tên (Req 6.5) ---
    clips, conflicts = plan_outputs(seg_results)
    manual_review_ids: set[str] = set()
    for conflict in conflicts:
        log.error(
            "Xung đột tên đầu ra %s giữa các video_id %s — đánh dấu rà soát thủ công.",
            conflict.out_filename,
            ", ".join(conflict.video_ids),
        )
        manual_review_ids.update(conflict.video_ids)

    # --- Bước 5: ghi từng clip (split_variants / split_variant_predicted / manual) ---
    # Clip tự tin ghi vào split_variants/. Clip dự đoán ghi vào split_variant_predicted/.
    # Video manual_review giữ bản gốc ở manual/ và thêm preview clip cắt sẵn vào
    # manual/review_clips/<video_id>/ để reviewer mở nhanh cả hai ngữ cảnh.
    seg_by_id = {r.video_id: r for r in seg_results}
    sub_clips_written = 0
    label_clips: list[OutputClip] = list(clips)
    if not dry_run:
        label_clips = []
        output_jobs: list[OutputWriteJob] = []
        for clip in clips:
            entry = discovered_ids[clip.video_id]
            props = props_by_id[clip.video_id]
            seg = seg_by_id[clip.video_id]
            output_jobs.append(OutputWriteJob(entry.path, clip, seg, props, cfg))

        result_slots: list[OutputWriteResult | None] = [None] * len(output_jobs)
        pending_jobs: list[OutputWriteJob] = []
        pending_indices: list[int] = []
        for index, output_job in enumerate(output_jobs):
            resumed = (
                _resume_output_result(output_job)
                if cfg.resume_existing
                else None
            )
            if resumed is None:
                pending_jobs.append(output_job)
                pending_indices.append(index)
            else:
                result_slots[index] = resumed

        log.info(
            "Bắt đầu ghi/tách view: total=%d resume=%d pending=%d, dùng %d process.",
            len(output_jobs),
            len(output_jobs) - len(pending_jobs),
            len(pending_jobs),
            int(cfg.parallel_workers),
        )
        fresh_results = _run_output_jobs(pending_jobs, int(cfg.parallel_workers))
        for index, fresh_result in zip(pending_indices, fresh_results):
            result_slots[index] = fresh_result
        output_results = [result for result in result_slots if result is not None]
        for output_result in output_results:
            for level, message in output_result.messages:
                getattr(log, level, log.info)(message)
            sub_clips_written += output_result.written
            label_clips.extend(output_result.label_clips)
            if output_result.manual_review:
                manual_review_ids.add(output_result.video_id)

    for seg in seg_results:
        if seg.kind == "manual_review":
            manual_review_ids.add(seg.video_id)

    if not dry_run and manual_review_ids:
        preview_root = cfg.manual_review_path / "review_clips"
        preview_root.mkdir(parents=True, exist_ok=True)
        copied = _copy_review_videos(
            manual_review_ids, discovered_ids, cfg.manual_review_path, "manual_review", log
        )
        preview_written = _write_review_preview_clips(
            manual_review_ids,
            discovered_ids,
            props_by_id,
            seg_by_id,
            cfg,
            detector,
            preview_root,
            log,
        )
        log.info(
            "manual: %d video_id cần review; copied=%d originals ở %s; preview clips=%d ở %s",
            len(manual_review_ids),
            copied,
            cfg.manual_review_path,
            preview_written,
            preview_root,
        )

    # --- Bước 6: dựng + ghi bảng nhãn mới (Req 7) ---
    # Dùng label_clips (clip thật đã ghi vào split_variants/, gồm cả _side) thay vì
    # clips gốc, để nhãn phản ánh đúng các file góc đã tách.
    if not dry_run:
        source_rows = read_source_labels(cfg)
        new_rows = build_new_rows(source_rows, label_clips, discovered_ids.keys(), logger=log)
        write_new_labels(new_rows, cfg, logger=log)

    # --- Bước 7: kế toán + ghi RunReport (Req 8.2) ---
    report = build_run_report(
        seg_results, skipped=skipped, manual_review_ids=manual_review_ids
    )
    log.info(
        "RunReport: total=%d single=%d multi=%d sub_clips=%d skipped=%d manual_review=%d",
        report.total_discovered,
        report.single_variant,
        report.multi_variant,
        report.sub_clips_generated,
        report.skipped,
        report.manual_review,
    )
    return report


def _make_default_detector(cfg: PreprocessConfig, log):
    """Create the lazy EasyOCR detector with the configured compute device."""
    from .number_detector import EasyOcrNumberDetector

    return EasyOcrNumberDetector(cfg=cfg, gpu=bool(cfg.ocr_gpu), log=log)


def _process_output_clip(job):
    """Expand and write one method clip inside a worker process."""
    clip = job.clip
    seg = job.seg
    props = job.props
    cfg = job.cfg
    log = _WorkerLog()

    # A pending job has no trustworthy completion marker. Remove only artifacts
    # belonging to this deterministic output base before rebuilding it.
    _clear_output_artifacts(job)

    span = _clip_span(clip, seg, props, cfg)
    if span is None:
        log.warning(
            "Cách %s của %s còn < 1 frame sau Biên_An_Toàn — bỏ qua + rà "
            "soát thủ công.",
            clip.variant_index,
            clip.video_id,
        )
        return OutputWriteResult(
            clip.video_id, 0, (), True, tuple(log.messages)
        )

    is_predicted = bool(getattr(seg, "inferred", False))
    base = clip.out_filename[: -len(".mp4")]
    view_jobs = _expand_views(job.src_path, base, span, props, cfg, log)
    written = 0
    manual_review = False
    label_clips: list[OutputClip] = []
    completed_outputs: list[dict[str, object]] = []

    for out_filename, sub_span, _is_side, route in view_jobs:
        if route == "manual":
            manual_review = True
            continue
        target_dir = (
            cfg.inferred_review_path
            if route == "predicted" or is_predicted
            else cfg.split_output_path
        )
        out_path = target_dir / out_filename
        if write_clip(job.src_path, out_path, sub_span, props, log):
            written += 1
            actual_route = (
                "predicted"
                if route == "predicted" or is_predicted
                else "confident"
            )
            completed_outputs.append(
                {
                    "filename": out_filename,
                    "route": actual_route,
                    "start_frame": int(sub_span.start_frame),
                    "end_frame": int(sub_span.end_frame),
                }
            )
            if route == "confident" and not is_predicted:
                label_clips.append(
                    OutputClip(
                        video_id=clip.video_id,
                        variant_index=clip.variant_index,
                        out_filename=out_filename,
                        start_frame=sub_span.start_frame,
                        end_frame=sub_span.end_frame,
                    )
                )
        else:
            manual_review = True

    if not manual_review and completed_outputs:
        try:
            _write_completion_manifest(job, completed_outputs)
        except OSError as exc:
            # Outputs remain usable now, but a future resume safely rebuilds them
            # because no atomic completion marker was published.
            log.warning("completion[%s]: could not write manifest: %s", base, exc)

    return OutputWriteResult(
        video_id=clip.video_id,
        written=written,
        label_clips=tuple(label_clips),
        manual_review=manual_review,
        messages=tuple(log.messages),
    )


def _output_is_readable(path: Path) -> bool:
    """Return whether an existing clip can be probed and contains frames."""
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    from .video_probe import probe

    props = probe(path)
    return props is not None and int(props.num_frames) > 0


def _completion_marker(job: OutputWriteJob) -> Path:
    base = job.clip.out_filename[: -len(".mp4")]
    return job.cfg.completion_path / f"{base}.json"


def _clear_output_artifacts(job: OutputWriteJob) -> None:
    """Remove stale files for exactly one deterministic output job."""
    base = job.clip.out_filename[: -len(".mp4")]
    names = (job.clip.out_filename, view_filename(base, is_side=True))
    for directory in (job.cfg.split_output_path, job.cfg.inferred_review_path):
        for name in names:
            try:
                (directory / name).unlink(missing_ok=True)
            except OSError:
                pass
    try:
        _completion_marker(job).unlink(missing_ok=True)
    except OSError:
        pass


def _write_completion_manifest(
    job: OutputWriteJob, outputs: list[dict[str, object]]
) -> None:
    """Atomically publish the complete output set for a resumable job."""
    marker = _completion_marker(job)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": OUTPUT_MANIFEST_VERSION,
        "video_id": job.clip.video_id,
        "variant_index": job.clip.variant_index,
        "outputs": outputs,
    }
    temporary = marker.with_name(f"{marker.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(marker)


def _resume_output_result(job: OutputWriteJob) -> OutputWriteResult | None:
    """Resume only an atomically completed and fully readable output set."""
    clip = job.clip
    base = clip.out_filename[: -len(".mp4")]
    marker = _completion_marker(job)
    if not marker.is_file():
        return None

    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        outputs = payload["outputs"]
        allowed_names = {clip.out_filename, view_filename(base, is_side=True)}
        if (
            payload.get("version") != OUTPUT_MANIFEST_VERSION
            or payload.get("video_id") != clip.video_id
            or payload.get("variant_index") != clip.variant_index
            or not isinstance(outputs, list)
            or not 1 <= len(outputs) <= 2
        ):
            raise ValueError("manifest identity or output count is invalid")

        seen_names: set[str] = set()
        resolved: list[tuple[dict[str, object], Path]] = []
        for item in outputs:
            if not isinstance(item, dict):
                raise ValueError("manifest output must be an object")
            filename = item.get("filename")
            route = item.get("route")
            if (
                filename not in allowed_names
                or filename in seen_names
                or route not in {"confident", "predicted"}
                or not isinstance(item.get("start_frame"), int)
                or not isinstance(item.get("end_frame"), int)
            ):
                raise ValueError("manifest output is invalid")
            seen_names.add(filename)
            directory = (
                job.cfg.split_output_path
                if route == "confident"
                else job.cfg.inferred_review_path
            )
            resolved.append((item, directory / str(filename)))

        if clip.out_filename not in seen_names:
            raise ValueError("manifest has no front output")
        if not all(_output_is_readable(path) for _, path in resolved):
            raise ValueError("manifest references a missing or unreadable output")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        try:
            marker.unlink(missing_ok=True)
        except OSError:
            pass
        return None

    labels = tuple(
        OutputClip(
            video_id=clip.video_id,
            variant_index=clip.variant_index,
            out_filename=path.name,
            start_frame=int(item["start_frame"]),
            end_frame=int(item["end_frame"]),
        )
        for item, path in resolved
        if item["route"] == "confident"
    )
    return OutputWriteResult(
        video_id=clip.video_id,
        written=0,
        label_clips=labels,
        manual_review=False,
        messages=(("info", f"resume[{base}]: dùng lại {len(resolved)} clip hợp lệ."),),
    )


def _run_output_jobs(jobs, max_workers: int, *, executor_factory=None):
    """Run independent clip jobs sequentially or in a bounded process pool."""
    if int(max_workers) <= 1:
        return [_process_output_clip(job) for job in jobs]

    if executor_factory is None:
        from concurrent.futures import ProcessPoolExecutor

        executor = ProcessPoolExecutor(
            max_workers=int(max_workers), initializer=_init_output_worker
        )
    else:
        executor = executor_factory(max_workers=int(max_workers))

    with executor:
        return list(executor.map(_process_output_clip, jobs, chunksize=1))


def _init_output_worker() -> None:
    """Prevent nested native thread pools from oversubscribing each process."""
    import os

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:  # noqa: BLE001 - worker can still run without this hint
        pass


def _copy_review_videos(video_ids, discovered_ids, dest_dir, label, log) -> int:
    """Copy file gốc của mỗi ``video_id`` vào *dest_dir* để con người xem lại.

    Dùng chung cho cả ``manual_review`` (cắt tay) lẫn ``inferred_review`` (tách
    bằng suy luận, cần kiểm ranh giới). File được **copy** (không move/hardlink)
    để giữ nguyên video nguồn và để folder tự chứa, dễ mở xem lại. Thư mục được
    tạo nếu thiếu. Lỗi copy từng file được ghi log và bỏ qua (không làm hỏng cả
    lần chạy); chỉ video đã thực sự duyệt được (có trong ``discovered_ids``) mới
    copy được.

    Args:
        video_ids: Tập ``video_id`` cần copy.
        discovered_ids: Mapping ``video_id -> VideoEntry`` (để lấy đường dẫn gốc).
        dest_dir: Thư mục đích (đường dẫn tuyệt đối trên ``D:``).
        label: Nhãn để ghi log ("manual_review" / "inferred_review").
        log: Logger của lần chạy.

    Returns:
        Số file đã copy thành công.
    """
    import shutil

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.error("Không tạo được thư mục %s %s: %s", label, dest_dir, exc)
        return 0

    copied = 0
    for video_id in sorted(video_ids):
        entry = discovered_ids.get(video_id)
        if entry is None:
            # Video không nằm trong tập đã duyệt (vd bị loại từ discovery) → bỏ qua.
            log.warning(
                "%s: không tìm thấy video gốc cho video_id=%s — bỏ qua copy.",
                label,
                video_id,
            )
            continue
        dest = dest_dir / f"{video_id}.mp4"
        try:
            shutil.copy2(entry.path, dest)
            copied += 1
        except OSError as exc:
            log.error("%s: lỗi copy %s -> %s: %s", label, entry.path, dest, exc)
    return copied


def _best_effort_preview_segmentation(entry, props, detector, cfg: PreprocessConfig, log):
    """Thử nhiều chế độ ranh giới để lấy preview clip cho video cần review."""
    methods: list[str] = []
    for method in ("ensemble", getattr(cfg, "boundary_method", "ocr"), "pose", "ocr"):
        if method not in methods:
            methods.append(method)

    for method in methods:
        preview_cfg = replace(cfg, boundary_method=method)
        preview = segment_video(entry, props, detector, preview_cfg, log)
        if preview.kind != "manual_review":
            return preview
    return None


def _write_review_preview_clips(
    video_ids,
    discovered_ids,
    props_by_id,
    seg_by_id,
    cfg: PreprocessConfig,
    detector,
    preview_root: Path,
    log,
) -> int:
    """Ghi preview clip cho video cần review vào manual/review_clips/<video_id>/."""
    written = 0
    for video_id in sorted(video_ids):
        entry = discovered_ids.get(video_id)
        props = props_by_id.get(video_id)
        seg = seg_by_id.get(video_id)
        if entry is None or props is None or seg is None:
            continue

        preview_seg = seg
        if preview_seg.kind == "manual_review":
            if detector is None:
                continue
            preview_seg = _best_effort_preview_segmentation(entry, props, detector, cfg, log)
            if preview_seg is None:
                continue

        clips, conflicts = plan_outputs([preview_seg])
        if conflicts or not clips:
            continue

        review_dir = preview_root / video_id
        review_dir.mkdir(parents=True, exist_ok=True)
        for clip in clips:
            span = VariantSpan(
                variant_index=clip.variant_index or 1,
                start_frame=clip.start_frame,
                end_frame=clip.end_frame,
            )
            out_path = review_dir / clip.out_filename
            if write_clip(entry.path, out_path, span, props, log):
                written += 1
    return written


@lru_cache(maxsize=8)
def _read_view_inventory(path_text: str) -> frozenset[str]:
    """Read canonical output IDs from a small CSV once per worker process."""
    with Path(path_text).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "video_id" not in reader.fieldnames:
            raise ValueError("view inventory CSV must contain a video_id column")
        return frozenset(
            str(row["video_id"]).strip()
            for row in reader
            if row.get("video_id") and str(row["video_id"]).strip()
        )


def _inventory_expects_side(base_name: str, cfg, log) -> bool | None:
    """Return True/False when an authoritative output inventory is configured."""
    inventory_path = getattr(cfg, "view_inventory_full_path", None)
    if inventory_path is None:
        return None
    try:
        output_ids = _read_view_inventory(str(inventory_path))
    except (OSError, ValueError, csv.Error) as exc:
        log.warning("view inventory %s is unreadable: %s", inventory_path, exc)
        return None
    return f"{base_name}_side" in output_ids


def _expand_views(src_path, base_name, span, props, cfg, log):
    """Tách MỘT đoạn cách thành (front, side) với pose-assisted confidence.

    Trả về danh sách ``(out_filename, VariantSpan, is_side, route)`` trong đó
    ``route ∈ {"confident", "predicted", "manual"}``.

    * Không có điểm cắt (một góc) → một phần tử: ``(<base>.mp4, span, False)``.
    * Có điểm cắt ``b`` trong ``[span.start, span.end]`` và cả hai đoạn còn >= 1
      frame → hai phần tử: front ``[span.start, b-1]`` giữ tên ``<base>.mp4``; side
      ``[b, span.end]`` thêm hậu tố ``<base>_side.mp4`` (METHOD-MAJOR, front trước).
    """
    from .segmenter import VariantSpan
    from .pose_boundary import detect_method_boundaries

    expected_side = _inventory_expects_side(base_name, cfg, log)
    if expected_side is False:
        # The inventory describes output topology only; no cut timestamp leaks
        # into detection. Avoid expensive pose scanning for known front-only clips.
        return [(view_filename(base_name, is_side=False), span, False, "confident")]

    hardcut = detect_hardcut(
        src_path,
        props,
        cfg,
        start_frame=span.start_frame,
        end_frame=span.end_frame,
        logger_obj=log,
    )
    pose_cuts = detect_method_boundaries(
        src_path,
        props,
        cfg,
        start_frame=span.start_frame,
        end_frame=span.end_frame,
        logger_obj=log,
    )
    pose_cut = pose_cuts[0] if pose_cuts else None

    if hardcut is None and pose_cut is None:
        if expected_side is True:
            log.warning(
                "_expand_views[%s]: inventory expects _side but no cut signal exists.",
                base_name,
            )
            return [(view_filename(base_name, is_side=False), span, False, "manual")]
        return [(view_filename(base_name, is_side=False), span, False, "confident")]

    # Tín hiệu confidence: đồng thuận lỏng giữa hard-cut và pose, và span đủ dài.
    min_frames = max(1, round(float(cfg.min_variant_seconds) * float(props.fps)))

    def _valid_cut(cut: int | None) -> bool:
        return cut is not None and span.start_frame < cut <= span.end_frame

    valid_hardcut = hardcut if _valid_cut(hardcut) else None
    valid_pose = pose_cut if _valid_cut(pose_cut) else None

    strict_tolerance = max(
        1, round(float(cfg.view_ensemble_tolerance_seconds) * float(props.fps))
    )
    predicted_tolerance = max(
        strict_tolerance,
        round(float(cfg.view_predicted_tolerance_seconds) * float(props.fps)),
    )

    if valid_hardcut is not None and valid_pose is None:
        if expected_side is True:
            chosen_cut = int(valid_hardcut)
            route = "predicted"
            log.info(
                "_expand_views[%s]: inventory xác nhận _side; dùng hard-cut-only "
                "frame %d và chuyển vào predicted.",
                base_name,
                chosen_cut,
            )
        else:
            log.info(
                "_expand_views[%s]: bỏ qua hard-cut-only frame %d cho view; cần pose "
                "đồng thuận để tạo _side.",
                base_name,
                int(valid_hardcut),
            )
            return [(view_filename(base_name, is_side=False), span, False, "confident")]

    elif valid_pose is None:
        return [(view_filename(base_name, is_side=False), span, False, "confident")]

    elif valid_hardcut is not None:
        delta = abs(int(valid_hardcut) - int(valid_pose))
        if delta <= strict_tolerance:
            chosen_cut = int(valid_hardcut)
            route = "confident"
        elif delta <= predicted_tolerance:
            chosen_cut = int(valid_hardcut)
            route = "predicted"
            log.info(
                "_expand_views[%s]: hard-cut frame %d lệch pose frame %d = %d "
                "frame; tách vào predicted.",
                base_name,
                int(valid_hardcut),
                int(valid_pose),
                delta,
            )
        elif expected_side is True:
            nearby_hardcut = detect_hardcut_near(
                src_path,
                props,
                cfg,
                reference_frame=int(valid_pose),
                radius_frames=predicted_tolerance,
                start_frame=span.start_frame,
                end_frame=span.end_frame,
                logger_obj=log,
            )
            chosen_cut = (
                int(nearby_hardcut)
                if _valid_cut(nearby_hardcut)
                else int(valid_hardcut)
            )
            route = "predicted"
            log.info(
                "_expand_views[%s]: inventory xác nhận _side; dùng pixel cut frame %d "
                "gần pose frame %d thay cho global hard-cut lệch %d frame.",
                base_name,
                chosen_cut,
                int(valid_pose),
                delta,
            )
        else:
            log.info(
                "_expand_views[%s]: bỏ qua hard-cut frame %d vì lệch pose frame %d "
                "quá %d frame.",
                base_name,
                int(valid_hardcut),
                int(valid_pose),
                predicted_tolerance,
            )
            return [(view_filename(base_name, is_side=False), span, False, "confident")]
    else:
        weak_hardcut = detect_hardcut_candidate(
            src_path,
            props,
            cfg,
            start_frame=span.start_frame,
            end_frame=span.end_frame,
            logger_obj=log,
        )
        valid_weak_hardcut = weak_hardcut if _valid_cut(weak_hardcut) else None
        if (
            valid_weak_hardcut is None
            or abs(int(valid_weak_hardcut) - int(valid_pose)) > predicted_tolerance
        ):
            if expected_side is True:
                log.warning(
                    "_expand_views[%s]: inventory expects _side but pose frame %d "
                    "has no nearby pixel cut.",
                    base_name,
                    int(valid_pose),
                )
                return [(view_filename(base_name, is_side=False), span, False, "manual")]
            log.info(
                "_expand_views[%s]: bỏ qua pose-only frame %d; không có ứng viên "
                "pixel gần trong %d frame.",
                base_name,
                int(valid_pose),
                predicted_tolerance,
            )
            return [(view_filename(base_name, is_side=False), span, False, "confident")]
        chosen_cut = int(valid_weak_hardcut)
        route = "predicted"
        log.info(
            "_expand_views[%s]: pose frame %d đồng thuận với pixel yếu frame %d; "
            "tách vào predicted.",
            base_name,
            int(valid_pose),
            int(valid_weak_hardcut),
        )

    front = VariantSpan(span.variant_index, span.start_frame, chosen_cut - 1)
    side = VariantSpan(span.variant_index, chosen_cut, span.end_frame)
    if front.end_frame < front.start_frame or side.end_frame < side.start_frame:
        return [(view_filename(base_name, is_side=False), span, False, "manual")]

    if min(front.end_frame - front.start_frame + 1, side.end_frame - side.start_frame + 1) < min_frames:
        log.warning(
            "_expand_views[%s]: span dưới %d frame sau hard-cut/pose → predicted/manual.",
            base_name,
            min_frames,
        )
        route = "predicted"

    log.info(
        "_expand_views[%s]: tách góc tại frame %d → %s + %s.",
        base_name,
        chosen_cut,
        view_filename(base_name, is_side=False),
        view_filename(base_name, is_side=True),
    )
    return [
        (view_filename(base_name, is_side=False), front, False, route),
        (view_filename(base_name, is_side=True), side, True, route),
    ]


def _clip_span(clip, seg, props, cfg):
    """Tính :class:`VariantSpan` đã trừ Biên_An_Toàn cho *clip*.

    Với video một cách (``variant_index is None``) trả về span phủ toàn bộ video.
    Với video nhiều cách, áp :func:`trimmed_spans` lên toàn bộ span của video rồi
    chọn span ứng với ``clip.variant_index``; trả ``None`` nếu span đó bị thu còn
    ``< 1`` frame (Req 5.8).
    """
    from .segmenter import VariantSpan

    num_frames = int(props.num_frames)

    if clip.variant_index is None:
        # Video một cách: giữ nguyên toàn bộ frame (không cắt — Req 5.4).
        return VariantSpan(variant_index=1, start_frame=0, end_frame=max(0, num_frames - 1))

    trimmed = trimmed_spans(seg.spans, num_frames, cfg.safety_margin_frames)
    for span in trimmed:
        if span.variant_index == clip.variant_index:
            return span
    return None


def _build_config_from_args(args) -> PreprocessConfig:
    """Dựng :class:`PreprocessConfig` từ các tham số CLI (chỉ ghi đè khi được cấp)."""
    project_root = Path(args.project_root) if args.project_root else Path.cwd()
    overrides = {}
    if args.sample_interval is not None:
        overrides["sample_interval_seconds"] = args.sample_interval
    if args.safety_margin is not None:
        overrides["safety_margin_frames"] = args.safety_margin
    if args.ocr_threshold is not None:
        overrides["ocr_confidence_threshold"] = args.ocr_threshold
    return PreprocessConfig(project_root=project_root, **overrides)


def main(argv=None) -> int:
    """Điểm vào CLI. Trả về mã thoát (0 = thành công)."""
    parser = argparse.ArgumentParser(
        prog="qipedc_video_preprocess.preprocess",
        description="Tách video QIPEDC nhiều cách thành các video con (một cách/clip).",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Thư mục gốc dự án trên ổ D: (mặc định: thư mục hiện tại).",
    )
    parser.add_argument("--sample-interval", type=float, default=None,
                        help="Khoảng lấy mẫu frame (giây, 0.1–5.0).")
    parser.add_argument("--safety-margin", type=int, default=None,
                        help="Biên an toàn quanh ranh giới (frame, 0–60).")
    parser.add_argument("--ocr-threshold", type=float, default=None,
                        help="Ngưỡng tin cậy OCR (0.0–1.0).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ duyệt + phân đoạn, không ghi clip/nhãn.")
    args = parser.parse_args(argv)

    cfg = _build_config_from_args(args)
    report = run_preprocess(cfg, dry_run=args.dry_run)

    # Mã thoát 0 luôn (lỗi cục bộ không làm cả lần chạy fail); báo tóm tắt ra stdout.
    print(
        f"Done. total={report.total_discovered} single={report.single_variant} "
        f"multi={report.multi_variant} sub_clips={report.sub_clips_generated} "
        f"skipped={report.skipped} manual_review={report.manual_review}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

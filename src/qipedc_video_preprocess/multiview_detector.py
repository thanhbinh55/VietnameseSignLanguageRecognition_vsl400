"""Bộ_Phát_Hiện_Đa_Góc — tìm điểm cắt cứng front→side bằng chênh lệch pixel.

Tích hợp ý tưởng của ``find_cut_and_put`` (github.com/luuluyenmotdongfor/
find_cut_and_put): một số video QIPEDC ghép HAI góc quay nối tiếp nhau theo thời
gian — góc CHÍNH DIỆN (front) trước, rồi CẮT CỨNG sang góc BÊN (side). Điểm cắt
cứng tạo một đỉnh chênh lệch pixel rất lớn giữa hai frame liền kề. Module này quét
frame, đo ``cv2.absdiff`` trung bình giữa các frame liên tiếp, và trả về frame nơi
đỉnh nổi bật rõ nhất = điểm chuyển góc.

Quy ước (khớp find_cut_and_put):

* Góc trước (front) LUÔN là đoạn ĐẦU; đoạn sau điểm cắt là góc bên (side).
* Trả ``None`` nếu không có đỉnh đủ rõ → video một-góc (không tách góc).

Hàm hoạt động trên một KHOẢNG frame ``[start_frame, end_frame]`` để ghép được với
bước tách-nhiều-cách theo thứ tự METHOD-MAJOR: với mỗi đoạn cách, dò điểm cắt góc
riêng trong đoạn đó.

Đây là lớp BIÊN (đọc frame + cv2). Logic ngưỡng/đỉnh tách riêng trong
:func:`pick_hardcut` để property-test được mà không cần video thật.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from .config import PreprocessConfig
    from .video_probe import VideoProps


def pick_hardcut(
    diffs: "list[float]",
    boundary_frames: "list[int]",
    span_start: int,
    span_end: int,
    min_gap: int,
    diff_threshold: float,
    prominence_ratio: float = 4.0,
) -> int | None:
    """Chọn điểm cắt cứng từ chuỗi chênh lệch (LOGIC THUẦN).

    Args:
        diffs: Chênh lệch pixel trung bình chuẩn hóa [0,1] giữa các mẫu liên tiếp.
        boundary_frames: ``boundary_frames[i]`` là frame BẮT ĐẦU đoạn-sau của cặp
            mẫu thứ ``i`` (ứng với ``diffs[i]``) — chính là ứng viên điểm cắt.
        span_start: Frame đầu của khoảng đang xét (để áp ``min_gap`` ở biên trái).
        span_end: Frame cuối của khoảng đang xét (biên phải).
        min_gap: Khoảng cách tối thiểu (frame) từ điểm cắt tới hai biên.
        diff_threshold: Ngưỡng tuyệt đối tối thiểu của đỉnh [0,1].
        prominence_ratio: Đỉnh phải >= ``prominence_ratio`` × nền (trung vị phần
            còn lại) để được chấp nhận — loại nhiễu nhỏ rải đều.

    Returns:
        Frame điểm cắt cứng (0-based), hoặc ``None`` nếu không có đỉnh đủ rõ.
    """
    if not diffs:
        return None

    peak_pos = -1
    peak_val = 0.0
    for i, d in enumerate(diffs):
        bf = boundary_frames[i]
        if bf < span_start + min_gap or bf > span_end - min_gap:
            continue
        if d > peak_val:
            peak_val = d
            peak_pos = i

    if peak_pos < 0:
        return None

    others = [d for j, d in enumerate(diffs) if j != peak_pos]
    baseline = _median(others) if others else 0.0

    if peak_val < diff_threshold:
        return None
    if baseline > 0.0 and peak_val < prominence_ratio * baseline:
        return None

    return boundary_frames[peak_pos]


def pick_fine_hardcut(
    diffs: "list[float]", boundary_frames: "list[int]"
) -> int | None:
    """Return the exact consecutive-frame boundary with the largest change."""
    if not diffs or len(diffs) != len(boundary_frames):
        return None
    return int(boundary_frames[max(range(len(diffs)), key=diffs.__getitem__)])


def _median(values: "list[float]") -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _detect_hardcut(
    video_path,
    props: "VideoProps",
    cfg: "PreprocessConfig",
    *,
    start_frame: int = 0,
    end_frame: "int | None" = None,
    logger_obj: "logging.Logger | None" = None,
    prominence_ratio: float = 4.0,
    min_gap_frames: "int | None" = None,
) -> int | None:
    """Định vị điểm cắt cứng front→side trong ``[start_frame, end_frame]`` (BIÊN).

    Quét frame trong khoảng, đo ``cv2.absdiff`` trung bình (ảnh xám thu nhỏ để
    nhanh) giữa các mẫu liên tiếp, rồi gọi :func:`pick_hardcut` để chọn đỉnh.

    Args:
        video_path: Đường dẫn video nguồn (mở capture riêng).
        props: :class:`VideoProps` (fps, num_frames).
        cfg: :class:`PreprocessConfig` — dùng ``multiview_diff_threshold`` và
            ``multiview_min_gap_seconds``.
        start_frame: Frame đầu khoảng xét (0-based, inclusive).
        end_frame: Frame cuối khoảng xét (inclusive); mặc định ``num_frames - 1``.
        logger_obj: Logger tùy chọn.

    Returns:
        Frame điểm cắt cứng (0-based) là điểm BẮT ĐẦU góc side; ``None`` nếu không
        phát hiện (một góc).
    """
    log = logger_obj if logger_obj is not None else logger
    if not getattr(cfg, "multiview_enabled", True):
        return None

    fps = float(props.fps)
    num_frames = int(props.num_frames)
    lo = max(0, int(start_frame))
    hi = num_frames - 1 if end_frame is None else min(int(end_frame), num_frames - 1)
    if fps <= 0 or hi - lo < 3:
        return None

    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            log.warning("detect_hardcut: không mở được video %s.", video_path)
            return None

        # Lấy mẫu ~10 frame/giây để bắt được điểm cắt mà không quét toàn bộ.
        step = max(1, round(fps * 0.1))
        indices = list(range(lo, hi + 1, step))
        if len(indices) < 3:
            return None

        small = (64, 64)  # thu nhỏ để đo chênh lệch toàn cục nhanh, ổn định
        prev = None
        diffs: list[float] = []
        boundary_frames: list[int] = []
        for idx in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            grabbed, frame = capture.read()
            if not grabbed or frame is None:
                prev = None
                continue
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, small, interpolation=cv2.INTER_AREA)
            except Exception:  # noqa: BLE001 - frame lỗi → bỏ mẫu
                prev = None
                continue
            if prev is not None:
                diffs.append(float(np.mean(cv2.absdiff(prev, gray))) / 255.0)
                boundary_frames.append(idx)  # điểm cắt = frame đoạn-sau
            prev = gray

        min_gap = (
            max(1, round(fps * float(cfg.multiview_min_gap_seconds)))
            if min_gap_frames is None
            else max(1, int(min_gap_frames))
        )
        boundary = pick_hardcut(
            diffs,
            boundary_frames,
            span_start=lo,
            span_end=hi,
            min_gap=min_gap,
            diff_threshold=float(cfg.multiview_diff_threshold),
            prominence_ratio=float(prominence_ratio),
        )
        if boundary is not None:
            boundary = _refine_hardcut(
                capture,
                boundary,
                step,
                lo,
                hi,
                cv2=cv2,
                np=np,
            )
            log.info(
                "detect_hardcut: điểm cắt góc tại frame %d trong [%d,%d].",
                boundary,
                lo,
                hi,
            )
        return boundary
    finally:
        capture.release()


def _refine_hardcut(capture, coarse_boundary, step, lo, hi, *, cv2, np) -> int:
    """Refine a sampled cut by checking every adjacent frame in its interval."""
    start = max(int(lo), int(coarse_boundary) - max(1, int(step)))
    end = min(int(hi), int(coarse_boundary))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    grabbed, frame = capture.read()
    if not grabbed or frame is None:
        return int(coarse_boundary)

    small = (64, 64)
    try:
        previous = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            small,
            interpolation=cv2.INTER_AREA,
        )
    except Exception:  # noqa: BLE001 - keep the safe coarse boundary
        return int(coarse_boundary)

    diffs: list[float] = []
    boundary_frames: list[int] = []
    for frame_index in range(start + 1, end + 1):
        grabbed, frame = capture.read()
        if not grabbed or frame is None:
            break
        try:
            current = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                small,
                interpolation=cv2.INTER_AREA,
            )
        except Exception:  # noqa: BLE001 - skip a corrupt frame
            continue
        diffs.append(float(np.mean(cv2.absdiff(previous, current))) / 255.0)
        boundary_frames.append(frame_index)
        previous = current

    return pick_fine_hardcut(diffs, boundary_frames) or int(coarse_boundary)


def detect_hardcut(
    video_path,
    props: "VideoProps",
    cfg: "PreprocessConfig",
    *,
    start_frame: int = 0,
    end_frame: "int | None" = None,
    logger_obj: "logging.Logger | None" = None,
) -> int | None:
    """Return a prominent pixel hard-cut suitable for a confident view split."""
    return _detect_hardcut(
        video_path,
        props,
        cfg,
        start_frame=start_frame,
        end_frame=end_frame,
        logger_obj=logger_obj,
        prominence_ratio=4.0,
    )


def detect_hardcut_candidate(
    video_path,
    props: "VideoProps",
    cfg: "PreprocessConfig",
    *,
    start_frame: int = 0,
    end_frame: "int | None" = None,
    logger_obj: "logging.Logger | None" = None,
) -> int | None:
    """Return the strongest absolute-threshold pixel candidate.

    This deliberately skips the prominence test and is only safe when another
    independent signal (currently pose) agrees within the predicted tolerance.
    """
    return _detect_hardcut(
        video_path,
        props,
        cfg,
        start_frame=start_frame,
        end_frame=end_frame,
        logger_obj=logger_obj,
        prominence_ratio=0.0,
    )


def detect_hardcut_near(
    video_path,
    props: "VideoProps",
    cfg: "PreprocessConfig",
    *,
    reference_frame: int,
    radius_frames: int,
    start_frame: int = 0,
    end_frame: "int | None" = None,
    logger_obj: "logging.Logger | None" = None,
) -> int | None:
    """Find the strongest absolute-threshold pixel cut near another signal."""
    span_end = int(props.num_frames) - 1 if end_frame is None else int(end_frame)
    radius = max(1, int(radius_frames))
    return _detect_hardcut(
        video_path,
        props,
        cfg,
        start_frame=max(int(start_frame), int(reference_frame) - radius),
        end_frame=min(span_end, int(reference_frame) + radius),
        logger_obj=logger_obj,
        prominence_ratio=0.0,
        min_gap_frames=1,
    )

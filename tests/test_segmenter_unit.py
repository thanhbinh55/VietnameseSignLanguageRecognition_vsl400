"""Unit tests cho ``qipedc_video_preprocess.segmenter`` â€” xá»­ lÃ½ lá»—i detector.

Phá»§ **Requirement 4.8**: náº¿u Bá»™_PhÃ¡t_Hiá»‡n_Sá»‘ bÃ¡o lá»—i hoáº·c quÃ¡ thá»i gian khi xá»­ lÃ½
má»™t frame máº«u, THE Bá»™_PhÃ¢n_Äoáº¡n SHALL coi frame Ä‘Ã³ lÃ  **khÃ´ng cÃ³ sá»‘ há»£p lá»‡**, ghi
log lÃ½ do, vÃ  **tiáº¿p tá»¥c** vá»›i cÃ¡c frame máº«u cÃ²n láº¡i (khÃ´ng há»§y cáº£ láº§n cháº¡y).

Hai Ä‘iá»ƒm vÃ o Ä‘Æ°á»£c kiá»ƒm:

* :func:`segmenter.segment_video` â€” dÃ¹ng má»™t :class:`FakeNumberDetector` nÃ©m lá»—i
  trÃªn má»™t frame nháº¥t Ä‘á»‹nh vÃ  má»™t :class:`FakeCapture` thay cho
  ``cv2.VideoCapture`` (monkeypatch ``cv2.VideoCapture`` trong module segmenter)
  Ä‘á»ƒ khÃ´ng cáº§n video tháº­t.
* :func:`segmenter.refine_boundary` â€” dÃ¹ng má»™t **nguá»“n frame callable** (khÃ´ng cáº§n
  OpenCV) vá»›i detector nÃ©m lá»—i trÃªn Ä‘Ãºng má»™t chá»‰ sá»‘ frame Ä‘Æ°á»£c dÃ² tá»›i.

Má»i cáº£nh bÃ¡o Ä‘Æ°á»£c báº¯t báº±ng má»™t logger tháº­t gáº¯n :class:`CapturingHandler` Ä‘á»ƒ kháº³ng
Ä‘á»‹nh lÃ½ do lá»—i Ä‘Ã£ Ä‘Æ°á»£c ghi log.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from qipedc_video_preprocess import segmenter
from qipedc_video_preprocess.config import PreprocessConfig
from qipedc_video_preprocess.discovery import VideoEntry
from qipedc_video_preprocess.number_detector import DetectionResult
from qipedc_video_preprocess.video_probe import VideoProps

# project_root giáº£ láº­p náº±m trÃªn á»• D: (Req 1.3). KhÃ´ng cháº¡m Ä‘Ä©a: capture Ä‘Æ°á»£c
# monkeypatch vÃ  refine_boundary dÃ¹ng nguá»“n frame callable.
PROJECT_ROOT = Path("D:/projects/metadata_VSL")


# --------------------------------------------------------------------------- #
# Tiá»‡n Ã­ch test
# --------------------------------------------------------------------------- #
class CapturingHandler(logging.Handler):
    """Handler thu tháº­p má»i :class:`logging.LogRecord` Ä‘á»ƒ kiá»ƒm tra trong test."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        self.records.append(record)

    def messages(self) -> list[str]:
        return [r.getMessage() for r in self.records]

    def messages_at(self, level: int) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno == level]


@pytest.fixture
def logger_and_handler() -> tuple[logging.Logger, CapturingHandler]:
    """Logger tháº­t, cÃ´ láº­p, gáº¯n :class:`CapturingHandler`."""
    handler = CapturingHandler()
    logger = logging.getLogger(f"test_segmenter_unit.{id(handler)}")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, handler


class FakeNumberDetector:
    """Bá»™ phÃ¡t hiá»‡n sá»‘ giáº£ láº­p theo chá»‰ sá»‘ frame.

    *frame* truyá»n vÃ o ``detect`` chÃ­nh lÃ  **chá»‰ sá»‘ frame** (do
    :class:`FakeCapture` vÃ  nguá»“n frame callable tráº£ vá» chá»‰ sá»‘). Detector tra
    ``numbers`` Ä‘á»ƒ láº¥y giÃ¡ trá»‹ (``int`` hoáº·c ``None``); náº¿u chá»‰ sá»‘ náº±m trong
    ``raise_on`` thÃ¬ **nÃ©m lá»—i** Ä‘á»ƒ mÃ´ phá»ng lá»—i/timeout cá»§a OCR (Req 4.8).
    """

    def __init__(
        self,
        numbers: dict[int, int | None],
        raise_on: set[int] | None = None,
        exc: BaseException | None = None,
    ) -> None:
        self._numbers = numbers
        self._raise_on = set(raise_on or set())
        self._exc = exc or TimeoutError("OCR quÃ¡ thá»i gian")
        self.calls: list[int] = []

    def detect(self, frame) -> DetectionResult:
        index = int(frame)
        self.calls.append(index)
        if index in self._raise_on:
            raise self._exc
        number = self._numbers.get(index)
        confidence = 0.0 if number is None else 0.9
        return DetectionResult(number=number, confidence=confidence)


class FakeCapture:
    """Thay tháº¿ ``cv2.VideoCapture``: ``read`` tráº£ vá» chá»‰ sá»‘ frame hiá»‡n táº¡i.

    ``set(CAP_PROP_POS_FRAMES, idx)`` ghi nháº­n vá»‹ trÃ­; ``read`` tráº£ ``(True, idx)``
    nÃªn "frame" chÃ­nh lÃ  chá»‰ sá»‘ â€” Ä‘á»§ Ä‘á»ƒ :class:`FakeNumberDetector` Ã¡nh xáº¡ thÃ nh
    sá»‘. KhÃ´ng phá»¥ thuá»™c OpenCV/video tháº­t.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._pos = 0
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - khá»›p API cv2
        return True

    def set(self, prop, value) -> bool:  # noqa: A003 - khá»›p API cv2
        self._pos = int(value)
        return True

    def read(self):
        return True, self._pos

    def release(self) -> None:
        self.released = True


def make_cfg(**overrides) -> PreprocessConfig:
    """Dá»±ng ``PreprocessConfig`` tháº­t vá»›i project_root trÃªn D:.

    fps=1.0 + ``sample_interval_seconds=1.0`` cho bÆ°á»›c láº¥y máº«u = 1, nÃªn chá»‰ sá»‘
    frame máº«u trÃ¹ng ``range(0, num_frames)`` â€” dá»… suy luáº­n trong test.
    """
    params = dict(project_root=PROJECT_ROOT, sample_interval_seconds=1.0)
    params.update(overrides)
    return PreprocessConfig(**params)


def make_entry(video_id: str = "W00202") -> VideoEntry:
    return VideoEntry(
        video_id=video_id,
        path=Path(f"D:/projects/metadata_VSL/Dataset/raw_videos/{video_id}.mp4"),
        source_dir="Dataset/raw_videos",
    )


def make_props(num_frames: int, fps: float = 1.0) -> VideoProps:
    return VideoProps(
        fps=fps,
        num_frames=num_frames,
        length_seconds=round(num_frames / fps, 2),
        resolution=(1280, 720),
    )


# --------------------------------------------------------------------------- #
# segment_video â€” Req 4.8: lá»—i detector trÃªn 1 frame â†’ None + log + tiáº¿p tá»¥c
# --------------------------------------------------------------------------- #
def test_segment_video_detector_error_on_one_frame_treated_as_no_number(
    monkeypatch, logger_and_handler
):
    """Detector nÃ©m lá»—i trÃªn frame máº«u 0 â†’ coi nhÆ° khÃ´ng cÃ³ sá»‘, váº«n hoÃ n táº¥t.

    CÃ¡c frame cÃ²n láº¡i Ä‘á»u "khÃ´ng cÃ³ sá»‘" nÃªn video phÃ¢n loáº¡i lÃ  single; Ä‘iá»u quan
    trá»ng lÃ  láº§n cháº¡y KHÃ”NG bá»‹ há»§y bá»Ÿi lá»—i vÃ  frame lá»—i Ä‘Æ°á»£c ghi lÃ  ``None``.
    """
    import cv2

    logger, handler = logger_and_handler
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)

    # num_frames=4, fps=1, interval=1 -> sample indices [0,1,2,3].
    detector = FakeNumberDetector(
        numbers={0: None, 1: None, 2: None, 3: None},
        raise_on={0},
        exc=RuntimeError("OCR engine crashed"),
    )
    cfg = make_cfg(boundary_method="ocr")

    result = segmenter.segment_video(
        make_entry(), make_props(num_frames=4), detector, cfg, logger=logger
    )

    # Láº§n cháº¡y hoÃ n táº¥t (khÃ´ng nÃ©m) vÃ  frame lá»—i Ä‘Æ°á»£c coi lÃ  khÃ´ng cÃ³ sá»‘.
    assert result.observed_numbers == (None, None, None, None)
    assert result.kind == "single"
    assert result.variant_count == 1

    # Táº¥t cáº£ frame máº«u Ä‘á»u Ä‘Æ°á»£c thá»­ (tiáº¿p tá»¥c sau frame lá»—i).
    assert detector.calls == [0, 1, 2, 3]

    # LÃ½ do lá»—i Ä‘Æ°á»£c ghi log (Req 4.8), nháº¯c tá»›i frame lá»—i vÃ  loáº¡i lá»—i.
    warnings = handler.messages_at(logging.WARNING)
    frame0_warnings = [m for m in warnings if "frame 0" in m]
    assert len(frame0_warnings) == 1
    assert "RuntimeError" in frame0_warnings[0]
    assert "4.8" in frame0_warnings[0]


def test_segment_video_default_ensemble_keeps_plain_single_video(
    monkeypatch, logger_and_handler
):
    """Default ensemble must not turn a plain no-overlay single video into manual."""
    import cv2

    logger, _ = logger_and_handler
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)

    detector = FakeNumberDetector(
        numbers={0: None, 1: None, 2: None, 3: None},
    )
    cfg = make_cfg()

    result = segmenter.segment_video(
        make_entry(), make_props(num_frames=4), detector, cfg, logger=logger
    )

    assert result.observed_numbers == (None, None, None, None)
    assert result.kind == "single"
    assert result.variant_count == 1


def test_segment_video_default_ensemble_agreement_is_confident(
    monkeypatch, logger_and_handler
):
    """OCR + pose agreement keeps a multi-way split in the confident route."""
    import cv2

    logger, _ = logger_and_handler
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(segmenter, "detect_method_boundaries", lambda *a, **k: [2])

    detector = FakeNumberDetector(numbers={0: 1, 1: 1, 2: 2, 3: 2})
    cfg = make_cfg()

    result = segmenter.segment_video(
        make_entry(), make_props(num_frames=4), detector, cfg, logger=logger
    )

    assert result.kind == "multi"
    assert result.variant_count == 2
    assert result.spans[0].end_frame == 1
    assert result.spans[1].start_frame == 2
    assert result.inferred is False


def test_segment_video_default_ensemble_half_second_drift_is_confident(
    monkeypatch, logger_and_handler
):
    """OCR and pose may drift up to the 0.6s default tolerance."""
    import cv2

    logger, _ = logger_and_handler
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(segmenter, "detect_method_boundaries", lambda *a, **k: [75])

    detector = FakeNumberDetector(
        numbers={frame: (1 if frame < 60 else 2) for frame in range(120)}
    )
    cfg = make_cfg(sample_interval_seconds=1.0)

    result = segmenter.segment_video(
        make_entry(), make_props(num_frames=120, fps=30.0), detector, cfg, logger=logger
    )

    assert result.kind == "multi"
    assert result.variant_count == 2
    assert result.spans[0].end_frame == 59
    assert result.spans[1].start_frame == 60
    assert result.inferred is False


def test_segment_video_ensemble_refines_every_predicted_method_boundary(
    monkeypatch, logger_and_handler
):
    """Regression for D0232: coarse 1-second samples must refine to number flashes."""
    import cv2

    logger, _ = logger_and_handler
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(segmenter, "detect_method_boundaries", lambda *a, **k: [])

    detector = FakeNumberDetector(
        numbers={
            frame: (1 if frame < 101 else 2 if frame < 220 else 3)
            for frame in range(320)
        }
    )
    cfg = make_cfg(sample_interval_seconds=1.0)

    result = segmenter.segment_video(
        make_entry("D0232"),
        make_props(num_frames=320, fps=30.0),
        detector,
        cfg,
        logger=logger,
    )

    assert result.kind == "multi"
    assert result.inferred is True
    assert [(s.start_frame, s.end_frame) for s in result.spans] == [
        (0, 100),
        (101, 219),
        (220, 319),
    ]


def test_segment_video_default_ensemble_ocr_only_is_predicted(
    monkeypatch, logger_and_handler
):
    """With default ensemble, one-sided OCR evidence is cut but routed to predicted."""
    import cv2

    logger, _ = logger_and_handler
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(segmenter, "detect_method_boundaries", lambda *a, **k: [])

    detector = FakeNumberDetector(numbers={0: 1, 1: 1, 2: 2, 3: 2})
    cfg = make_cfg()

    result = segmenter.segment_video(
        make_entry(), make_props(num_frames=4), detector, cfg, logger=logger
    )

    assert result.kind == "multi"
    assert result.variant_count == 2
    assert result.inferred is True


def test_segment_video_ensemble_extra_pose_boundaries_do_not_create_methods(
    monkeypatch, logger_and_handler
):
    """Pose may see view/action boundaries, but OCR/overlay owns method count."""
    import cv2

    logger, _ = logger_and_handler
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(segmenter, "detect_method_boundaries", lambda *a, **k: [122, 226, 364])

    detector = FakeNumberDetector(
        numbers={frame: (1 if frame < 226 else 2) for frame in range(486)}
    )
    cfg = make_cfg(sample_interval_seconds=1.0)

    result = segmenter.segment_video(
        make_entry("W00144"),
        make_props(num_frames=486, fps=30.0),
        detector,
        cfg,
        logger=logger,
    )

    assert result.kind == "multi"
    assert result.variant_count == 2
    assert [(s.variant_index, s.start_frame, s.end_frame) for s in result.spans] == [
        (1, 0, 225),
        (2, 226, 485),
    ]
    assert result.inferred is False


def test_ensemble_falls_back_to_pose_when_refined_ocr_boundary_collapses(
    monkeypatch, logger_and_handler
):
    """Regression for W00792: an implausible OCR cut must not discard a valid pose cut."""
    import cv2

    logger, _ = logger_and_handler
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(segmenter, "detect_method_boundaries", lambda *a, **k: [50])

    detector = FakeNumberDetector(
        numbers={frame: (1 if frame < 1 else 2) for frame in range(100)}
    )
    cfg = make_cfg(sample_interval_seconds=1.0)

    result = segmenter.segment_video(
        make_entry("W00792"),
        make_props(num_frames=100, fps=10.0),
        detector,
        cfg,
        logger=logger,
    )

    assert result.kind == "multi"
    assert result.inferred is True
    assert [(s.start_frame, s.end_frame) for s in result.spans] == [
        (0, 49),
        (50, 99),
    ]


def test_segment_video_continues_detecting_after_error(
    monkeypatch, logger_and_handler
):
    """Sau frame lá»—i, cÃ¡c frame cÃ²n láº¡i VáºªN Ä‘Æ°á»£c phÃ¡t hiá»‡n sá»‘ bÃ¬nh thÆ°á»ng.

    detect: 0->5, 1->lá»—i, 2->5, 3->5  =>  observed = (5, None, 5, 5). Chuá»—i sá»‘
    khÃ´ng báº¯t Ä‘áº§u tá»« 1 nÃªn phÃ¢n loáº¡i manual_review, nhÆ°ng Ä‘iá»ƒm cá»‘t lÃµi lÃ  chuá»—i
    quan sÃ¡t giá»¯ Ä‘Ãºng sá»‘ á»Ÿ cÃ¡c frame khÃ´ng lá»—i vÃ  ``None`` á»Ÿ frame lá»—i.
    """
    import cv2

    logger, handler = logger_and_handler
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)

    detector = FakeNumberDetector(
        numbers={0: 5, 1: 5, 2: 5, 3: 5},
        raise_on={1},
        exc=TimeoutError("OCR timeout"),
    )
    cfg = make_cfg(boundary_method="ocr")

    result = segmenter.segment_video(
        make_entry(), make_props(num_frames=4), detector, cfg, logger=logger
    )

    # Frame lá»—i -> None; cÃ¡c frame khÃ¡c giá»¯ sá»‘ Ä‘Ã£ Ä‘á»c.
    assert result.observed_numbers == (5, None, 5, 5)
    assert detector.calls == [0, 1, 2, 3]

    # Chuá»—i khÃ´ng báº¯t Ä‘áº§u tá»« 1 -> cáº§n rÃ  soÃ¡t thá»§ cÃ´ng (run váº«n hoÃ n táº¥t).
    assert result.kind == "manual_review"
    assert result.variant_count == 0

    frame1_warnings = [
        m for m in handler.messages_at(logging.WARNING) if "frame 1" in m
    ]
    assert len(frame1_warnings) == 1
    assert "TimeoutError" in frame1_warnings[0]
    assert "4.8" in frame1_warnings[0]


# --------------------------------------------------------------------------- #
# refine_boundary â€” Req 4.8: detector lá»—i trÃªn 1 chá»‰ sá»‘ Ä‘Æ°á»£c dÃ² â†’ log + tiáº¿p tá»¥c
# --------------------------------------------------------------------------- #
def test_refine_boundary_detector_error_logged_and_continues(logger_and_handler):
    """detect nÃ©m lá»—i trÃªn má»™t frame Ä‘Æ°á»£c dÃ² tá»›i â†’ coi nhÆ° khÃ´ng mang sá»‘ má»›i.

    Detector "Ä‘Æ¡n Ä‘iá»‡u": frame < 5 mang giÃ¡ trá»‹ cÅ© (1), frame >= 5 mang
    ``target_number`` (2), NHÆ¯NG nÃ©m lá»—i táº¡i frame 4. TÃ¬m-nhá»‹-phÃ¢n dÃ² cÃ¡c frame
    5, 3, 4; táº¡i frame 4 detector nÃ©m lá»—i â†’ bá»‹ coi lÃ  khÃ´ng mang sá»‘ má»›i (Ä‘Ãºng,
    vÃ¬ 4 < 5) â†’ ghi log vÃ  tiáº¿p tá»¥c, káº¿t quáº£ ranh giá»›i váº«n Ä‘Ãºng = 5.
    """
    logger, handler = logger_and_handler

    def detect_number(index: int) -> int:
        return 2 if index >= 5 else 1

    detector = FakeNumberDetector(
        numbers={i: detect_number(i) for i in range(0, 9)},
        raise_on={4},
        exc=RuntimeError("OCR engine crashed"),
    )

    # Nguá»“n frame callable: tráº£ vá» chÃ­nh chá»‰ sá»‘ (khÃ´ng bao giá» None).
    frame_source = lambda index: index  # noqa: E731

    refined = segmenter.refine_boundary(
        detector,
        frame_source,
        coarse_lo=2,
        coarse_hi=8,
        target_number=2,
        cfg=make_cfg(),
        logger=logger,
    )

    # Ranh giá»›i tinh chá»‰nh náº±m trong (coarse_lo, coarse_hi] vÃ  Ä‘Ãºng Ä‘iá»ƒm chuyá»ƒn.
    assert refined == 5

    # Lá»—i táº¡i frame 4 Ä‘Ã£ Ä‘Æ°á»£c dÃ² tá»›i vÃ  Ä‘Æ°á»£c ghi log (Req 4.8).
    assert 4 in detector.calls
    frame4_warnings = [
        m for m in handler.messages_at(logging.WARNING) if "frame 4" in m
    ]
    assert len(frame4_warnings) == 1
    assert "RuntimeError" in frame4_warnings[0]
    assert "4.8" in frame4_warnings[0]


def test_refine_boundary_unreadable_frame_logged_and_continues(logger_and_handler):
    """Nguá»“n frame tráº£ ``None`` táº¡i frame Ä‘Æ°á»£c dÃ² â†’ coi nhÆ° khÃ´ng cÃ³ sá»‘ + log.

    Cá»§ng cá»‘ nhÃ¡nh Ä‘á»c-frame cá»§a Req 4.8 trong :func:`refine_boundary`: frame 4
    khÃ´ng Ä‘á»c Ä‘Æ°á»£c (``None``) -> khÃ´ng mang sá»‘ má»›i (Ä‘Ãºng vÃ¬ 4 < 5) -> ghi log vÃ 
    tiáº¿p tá»¥c, ranh giá»›i váº«n Ä‘Ãºng = 5.
    """
    logger, handler = logger_and_handler

    detector = FakeNumberDetector(
        numbers={i: (2 if i >= 5 else 1) for i in range(0, 9)},
    )

    def frame_source(index: int):
        return None if index == 4 else index

    refined = segmenter.refine_boundary(
        detector,
        frame_source,
        coarse_lo=2,
        coarse_hi=8,
        target_number=2,
        cfg=make_cfg(),
        logger=logger,
    )

    assert refined == 5
    unreadable_warnings = [
        m
        for m in handler.messages_at(logging.WARNING)
        if "frame 4" in m and "khÃ´ng Ä‘á»c Ä‘Æ°á»£c" in m
    ]
    assert len(unreadable_warnings) == 1
    assert "4.8" in unreadable_warnings[0]


def test_result_from_boundaries_marks_short_confident_span_as_inferred():
    """A confident ensemble cut that creates a too-short span must be reviewed."""
    result = segmenter._result_from_boundaries(
        "W_SHORT",
        make_props(num_frames=30, fps=30.0),
        [10],
        observed_numbers=(1, 2),
        cfg=make_cfg(min_variant_seconds=0.6),
        inferred=False,
    )

    assert result is not None
    assert result.kind == "multi"
    assert result.inferred is True


def test_result_from_boundaries_rejects_short_inferred_span():
    """If the boundary was already inferred, a too-short span falls back to manual."""
    result = segmenter._result_from_boundaries(
        "W_SHORT",
        make_props(num_frames=30, fps=30.0),
        [10],
        observed_numbers=(1, 2),
        cfg=make_cfg(min_variant_seconds=0.6),
        inferred=True,
    )

    assert result is None

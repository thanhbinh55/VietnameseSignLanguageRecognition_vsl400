"""Bá»™_PhÃ¢n_Äoáº¡n â€” xÃ¡c Ä‘á»‹nh sá»‘ cÃ¡ch vÃ  ranh giá»›i giá»¯a cÃ¡c cÃ¡ch.

Module nÃ y Ä‘á»‹nh nghÄ©a cÃ¡c kiá»ƒu dá»¯ liá»‡u káº¿t quáº£ phÃ¢n Ä‘oáº¡n (:class:`VariantSpan`,
:class:`SegmentationResult`) vÃ  bá»™ sinh chá»‰ sá»‘ frame máº«u thuáº§n
(:func:`sample_frame_indices`).

Pháº§n logic phÃ¢n loáº¡i chuá»—i sá»‘ (``classify_sequence``), tinh chá»‰nh ranh giá»›i
(``refine_boundary``) vÃ  Ä‘iá»u phá»‘i (``segment_video``) sáº½ Ä‘Æ°á»£c hiá»‡n thá»±c á»Ÿ cÃ¡c
task sau (7.3 / 7.5). Xem design.md â€” Requirement 4.

Quy Æ°á»›c frame:

* Frame Ä‘Æ°á»£c Ä‘Ã¡nh chá»‰ sá»‘ **0-based**.
* :class:`VariantSpan` dÃ¹ng khoáº£ng **Ä‘Ã³ng** ``[start_frame, end_frame]``
  (``end_frame`` inclusive).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from .pose_boundary import detect_method_boundaries

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - chá»‰ phá»¥c vá»¥ type hint, trÃ¡nh import vÃ²ng
    from .config import PreprocessConfig
    from .discovery import VideoEntry
    from .number_detector import NumberDetector
    from .video_probe import VideoProps


@dataclass(frozen=True)
class VariantSpan:
    """Khoáº£ng frame cá»§a má»™t cÃ¡ch (variant) trong video.

    Khoáº£ng lÃ  **Ä‘Ã³ng** ``[start_frame, end_frame]`` (cáº£ hai Ä‘áº§u inclusive), frame
    Ä‘Ã¡nh chá»‰ sá»‘ 0-based. ``start_frame``/``end_frame`` lÃ  vá»‹ trÃ­ ranh giá»›i Ä‘Ã£ Ä‘Æ°á»£c
    tinh chá»‰nh vÃ  **chÆ°a** trá»« ``safety_margin`` (viá»‡c trá»« biÃªn an toÃ n do
    ``splitter.trimmed_spans`` Ä‘áº£m nhiá»‡m).

    Attributes:
        variant_index: Sá»‘ thá»© tá»± cÃ¡ch, 1-based, theo thá»© tá»± thá»i gian.
        start_frame: Frame báº¯t Ä‘áº§u (0-based, inclusive).
        end_frame: Frame káº¿t thÃºc (0-based, inclusive).
    """

    variant_index: int
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class SegmentationResult:
    """Káº¿t quáº£ phÃ¢n Ä‘oáº¡n cá»§a má»™t video.

    Attributes:
        video_id: ``video_id`` cá»§a video nguá»“n (stem), vÃ­ dá»¥ ``"W00202"``.
        kind: PhÃ¢n loáº¡i video â€” má»™t trong ``"single"``, ``"multi"`` hoáº·c
            ``"manual_review"``.
        variant_count: Sá»‘ cÃ¡ch â€” ``1`` náº¿u ``single``; ``N`` náº¿u ``multi``;
            ``0`` náº¿u ``manual_review``.
        spans: Tuple cÃ¡c :class:`VariantSpan` liÃªn tá»¥c khÃ´ng chá»“ng láº¥n theo thá»© tá»±
            thá»i gian (rá»—ng khi ``manual_review``).
        observed_numbers: Chuá»—i con sá»‘ quan sÃ¡t Ä‘Æ°á»£c trÃªn cÃ¡c frame máº«u (má»—i pháº§n
            tá»­ lÃ  ``int`` hoáº·c ``None`` cho "khÃ´ng cÃ³ sá»‘"), giá»¯ láº¡i Ä‘á»ƒ ghi log.
        inferred: ``True`` náº¿u phÃ¢n Ä‘oáº¡n multi nÃ y Ä‘Æ°á»£c dá»±ng nhá» **SUY LUáº¬N** (cÃ³
            nhÃ£n "CÃCH" + Ä‘iá»ƒm chuyá»ƒn sá»‘ rÃµ, nhÆ°ng OCR khÃ´ng Ä‘á»c trá»n dÃ£y
            ``1,2,â€¦``; dÃ£y Ä‘Æ°á»£c khÃ´i phá»¥c báº±ng rÃ ng buá»™c sá»‘ cÃ¡ch liÃªn tiáº¿p). CÃ¡c
            video nÃ y váº«n Ä‘Æ°á»£c tÃ¡ch tá»± Ä‘á»™ng NHÆ¯NG cáº§n con ngÆ°á»i kiá»ƒm láº¡i ranh giá»›i
            â†’ gom vÃ o folder ``inferred_review``. ``False`` cho má»i káº¿t quáº£ khÃ¡c.
    """

    video_id: str
    kind: str
    variant_count: int
    spans: tuple[VariantSpan, ...]
    observed_numbers: tuple[int | None, ...]
    inferred: bool = False


def sample_frame_indices(
    fps: float, num_frames: int, sample_interval_seconds: float
) -> list[int]:
    """Sinh cÃ¡c chá»‰ sá»‘ frame máº«u cÃ¡ch Ä‘á»u trong ``[0, num_frames)``.

    ÄÃ¢y lÃ  **logic thuáº§n** (khÃ´ng cháº¡m I/O) Ä‘á»ƒ phá»¥c vá»¥ property-test. Hiá»‡n thá»±c
    Req 4.1: láº¥y máº«u frame theo ``sample_interval_seconds``.

    BÆ°á»›c láº¥y máº«u Ä‘Æ°á»£c tÃ­nh báº±ng::

        step = max(1, round(fps * sample_interval_seconds))

    CÃ¡c chá»‰ sá»‘ báº¯t Ä‘áº§u tá»« ``0`` vÃ  tÄƒng dáº§n theo ``step`` trong khi cÃ²n nhá» hÆ¡n
    ``num_frames``. VÃ¬ ``step >= 1`` nÃªn dÃ£y chá»‰ sá»‘ luÃ´n **tÄƒng nghiÃªm ngáº·t** vÃ 
    náº±m trá»n trong khoáº£ng ``[0, num_frames)``.

    Args:
        fps: Tá»‘c Ä‘á»™ khung hÃ¬nh cá»§a video (khung/giÃ¢y).
        num_frames: Tá»•ng sá»‘ frame cá»§a video.
        sample_interval_seconds: Khoáº£ng thá»i gian láº¥y máº«u (giÃ¢y).

    Returns:
        Danh sÃ¡ch chá»‰ sá»‘ frame (0-based) tÄƒng nghiÃªm ngáº·t. Tráº£ vá» danh sÃ¡ch rá»—ng
        khi ``num_frames <= 0``.
    """
    if num_frames <= 0:
        return []

    step = max(1, round(fps * sample_interval_seconds))
    return list(range(0, num_frames, step))


# Loáº¡i Ä‘á»‘i tÆ°á»£ng máº«u: cáº·p (chá»‰_sá»‘_frame, sá»‘|None). ``None`` nghÄ©a lÃ  "khÃ´ng cÃ³ sá»‘"
# trÃªn frame máº«u Ä‘Ã³ (frame rá»—ng, OCR khÃ´ng Ä‘á»c Ä‘Æ°á»£c, hoáº·c detector bÃ¡o lá»—i â€” Req 4.8).
Sample = tuple[int, "int | None"]


def classify_sequence(
    samples: list[Sample],
    cfg: "PreprocessConfig",
    video_id: str = "",
    saw_cach_flags: "list[bool] | None" = None,
) -> SegmentationResult:
    """PhÃ¢n loáº¡i má»™t video thÃ nh single / multi / manual_review (LOGIC THUáº¦N).

    HÃ m chá»‰ lÃ m viá»‡c trÃªn chuá»—i quan sÃ¡t ``samples`` â€” danh sÃ¡ch cáº·p
    ``(frame_index, number | None)`` theo thá»© tá»± thá»i gian (frame_index tÄƒng dáº§n)
    â€” nÃªn cÃ³ thá»ƒ property-test mÃ  khÃ´ng cáº§n video tháº­t. ÄÃ¢y lÃ  káº¿t quáº£ **thÃ´**:
    cÃ¡c ranh giá»›i Ä‘áº·t táº¡i frame máº«u nÆ¡i giÃ¡ trá»‹ sá»‘ má»›i xuáº¥t hiá»‡n láº§n Ä‘áº§u; bÆ°á»›c
    tinh chá»‰nh Â±1 frame do :func:`refine_boundary` (task 7.5) Ä‘áº£m nhiá»‡m.

    Quy táº¯c phÃ¢n loáº¡i (Req 4.2/4.3/4.5/4.6/4.7):

    * **single** (``variant_count = 1``): khÃ´ng cÃ³ frame máº«u nÃ o mang sá»‘ há»£p lá»‡
      (má»i giÃ¡ trá»‹ lÃ  ``None``) â€” Req 4.5. ``spans`` gá»“m Ä‘Ãºng má»™t khoáº£ng phá»§ toÃ n
      bá»™ pháº¡m vi Ä‘Ã£ láº¥y máº«u.
    * **multi** (``variant_count = N``, ``N >= 2``): cÃ¡c giÃ¡ trá»‹ sá»‘ phÃ¢n biá»‡t theo
      thá»© tá»± thá»i gian táº¡o thÃ nh dÃ£y tÄƒng-liá»n-ká» báº¯t Ä‘áº§u tá»« 1 (``1, 2, â€¦, N``),
      **vÃ ** má»—i giÃ¡ trá»‹ má»›i (2..N) Ä‘Æ°á»£c xÃ¡c nháº­n trÃªn ``>= cfg.boundary_confirm_frames``
      frame máº«u **liÃªn tiáº¿p** ká»ƒ tá»« láº§n xuáº¥t hiá»‡n Ä‘áº§u â€” Req 4.2/4.6. Káº¿t quáº£ cÃ³
      Ä‘Ãºng ``N - 1`` ranh giá»›i; ``spans`` lÃ  ``N`` khoáº£ng liÃªn tá»¥c, khÃ´ng chá»“ng
      láº¥n, phá»§ kÃ­n pháº¡m vi Ä‘Ã£ láº¥y máº«u. Ranh giá»›i giá»¯a cÃ¡ch ``k`` vÃ  ``k+1`` náº±m
      táº¡i frame máº«u **Ä‘áº§u tiÃªn** mang giÃ¡ trá»‹ ``k+1``: cÃ¡ch ``k`` káº¿t thÃºc ngay
      trÆ°á»›c frame Ä‘Ã³, cÃ¡ch ``k+1`` báº¯t Ä‘áº§u táº¡i frame Ä‘Ã³.
    * **manual_review** (``variant_count = 0``, ``spans`` rá»—ng): má»i trÆ°á»ng há»£p cÃ²n
      láº¡i â€” khÃ´ng báº¯t Ä‘áº§u tá»« 1, nháº£y khÃ´ng liá»n ká» (vd 2â†’5), Ä‘áº£o thá»© tá»±, thiáº¿u sá»‘,
      chá»‰ cÃ³ Ä‘Ãºng má»™t giÃ¡ trá»‹ (vd chá»‰ tháº¥y "1" mÃ  khÃ´ng cÃ³ "2"), hoáº·c má»™t bÆ°á»›c
      chuyá»ƒn khÃ´ng Ä‘Æ°á»£c xÃ¡c nháº­n Ä‘á»§ sá»‘ frame liÃªn tiáº¿p â€” Req 4.3/4.7.

    ``observed_numbers`` luÃ´n giá»¯ **nguyÃªn** chuá»—i giÃ¡ trá»‹ quan sÃ¡t (ká»ƒ cáº£
    ``None``) theo Ä‘Ãºng thá»© tá»± Ä‘á»ƒ ghi log (Req 4.7/8.3).

    Args:
        samples: Danh sÃ¡ch cáº·p ``(frame_index, number | None)`` theo thá»© tá»± thá»i
            gian. ``frame_index`` lÃ  chá»‰ sá»‘ frame 0-based; ``number`` lÃ  sá»‘ nguyÃªn
            ``>= 1`` Ä‘Ã£ Ä‘áº¡t NgÆ°á»¡ng_Tin_Cáº­y, hoáº·c ``None`` cho "khÃ´ng cÃ³ sá»‘".
        cfg: :class:`~qipedc_video_preprocess.config.PreprocessConfig`; chá»‰ dÃ¹ng
            ``boundary_confirm_frames``.
        video_id: ``video_id`` cá»§a video nguá»“n (máº·c Ä‘á»‹nh rá»—ng; ``segment_video``
            sáº½ Ä‘iá»n giÃ¡ trá»‹ tháº­t á»Ÿ task 7.5).

    Returns:
        :class:`SegmentationResult` thÃ´.
    """
    observed_numbers: tuple[int | None, ...] = tuple(num for _, num in samples)
    confirm = max(1, int(cfg.boundary_confirm_frames))

    # CÃ³ frame nÃ o nhÃ¬n tháº¥y nhÃ£n "CÃCH" khÃ´ng? Náº¿u video cÃ³ overlay "CÃCH" mÃ  ta
    # KHÃ”NG dá»±ng Ä‘Æ°á»£c phÃ¢n Ä‘oáº¡n multi há»£p lá»‡ thÃ¬ Ä‘Ã³ lÃ  dáº¥u hiá»‡u nghi ngá» (sá»‘ Ä‘á»c
    # khÃ´ng cháº¯c) â†’ pháº£i rÃ  soÃ¡t thá»§ cÃ´ng thay vÃ¬ láº·ng láº½ coi lÃ  má»™t-cÃ¡ch (Req:
    # khÃ´ng Ã¢m tháº§m bá» sÃ³t video nhiá»u cÃ¡ch).
    any_cach = bool(saw_cach_flags) and any(saw_cach_flags)

    # KhÃ´ng cÃ³ frame máº«u nÃ o â†’ khÃ´ng cÃ³ gÃ¬ Ä‘á»ƒ phÃ¢n loáº¡i; coi nhÆ° single rá»—ng.
    if not samples:
        return SegmentationResult(
            video_id=video_id,
            kind="single",
            variant_count=1,
            spans=(),
            observed_numbers=observed_numbers,
        )

    lo = samples[0][0]
    hi = samples[-1][0]

    # Chuá»—i cÃ¡c láº§n "Ä‘á»•i giÃ¡ trá»‹" theo thá»i gian, bá» qua None (None = khÃ´ng quan
    # sÃ¡t, khÃ´ng phÃ¡ vá»¡ tÃ­nh liÃªn tá»¥c cá»§a má»™t giÃ¡ trá»‹). Má»—i pháº§n tá»­ ghi láº¡i
    # (giÃ¡_trá»‹, vá»‹_trÃ­_trong_samples, frame_index) táº¡i láº§n xuáº¥t hiá»‡n Ä‘áº§u cá»§a giÃ¡
    # trá»‹ Ä‘Ã³ trong má»™t block liÃªn tá»¥c.
    transitions: list[tuple[int, int, int]] = []
    last_value: int | None = None
    for pos, (frame_index, num) in enumerate(samples):
        if num is None:
            continue
        if num != last_value:
            transitions.append((num, pos, frame_index))
            last_value = num

    # Req 4.5: khÃ´ng cÃ³ sá»‘ há»£p lá»‡ nÃ o â†’ Video_Má»™t_CÃ¡ch.
    if not transitions:
        # NGOáº I Lá»†: náº¿u cÃ³ frame tháº¥y nhÃ£n "CÃCH" mÃ  khÃ´ng Ä‘á»c ná»•i sá»‘ nÃ o, Ä‘Ã¢y lÃ 
        # video cÃ³ overlay nhiá»u-cÃ¡ch nhÆ°ng OCR tháº¥t báº¡i â†’ rÃ  soÃ¡t thá»§ cÃ´ng, KHÃ”NG
        # coi lÃ  má»™t-cÃ¡ch (trÃ¡nh cáº¯t sÃ³t).
        if any_cach:
            return SegmentationResult(
                video_id=video_id,
                kind="manual_review",
                variant_count=0,
                spans=(),
                observed_numbers=observed_numbers,
            )
        return SegmentationResult(
            video_id=video_id,
            kind="single",
            variant_count=1,
            spans=(VariantSpan(variant_index=1, start_frame=lo, end_frame=hi),),
            observed_numbers=observed_numbers,
        )

    value_sequence = [value for value, _, _ in transitions]
    n_variants = len(value_sequence)

    # Multi há»£p lá»‡ âŸº dÃ£y giÃ¡ trá»‹ phÃ¢n biá»‡t Ä‘Ãºng báº±ng [1, 2, â€¦, N] vá»›i N >= 2.
    is_consecutive_from_one = value_sequence == list(range(1, n_variants + 1))

    def _confirmed(pos: int, value: int) -> bool:
        """Äáº¿m sá»‘ frame máº«u liÃªn tiáº¿p mang ``value`` ká»ƒ tá»« vá»‹ trÃ­ ``pos``."""
        run = 0
        for probe in range(pos, len(samples)):
            if samples[probe][1] == value:
                run += 1
            else:
                break
        return run >= confirm

    # Má»—i giÃ¡ trá»‹ má»›i (2..N) pháº£i Ä‘Æ°á»£c xÃ¡c nháº­n trÃªn >= confirm frame liÃªn tiáº¿p.
    all_confirmed = all(
        _confirmed(pos, value) for value, pos, _ in transitions[1:]
    )

    if n_variants >= 2 and is_consecutive_from_one and all_confirmed:
        boundary_frames = [frame_index for _, _, frame_index in transitions]
        spans: list[VariantSpan] = []
        for k in range(n_variants):
            start = lo if k == 0 else boundary_frames[k]
            end = (hi if k == n_variants - 1 else boundary_frames[k + 1] - 1)
            spans.append(
                VariantSpan(variant_index=k + 1, start_frame=start, end_frame=end)
            )
        return SegmentationResult(
            video_id=video_id,
            kind="multi",
            variant_count=n_variants,
            spans=tuple(spans),
            observed_numbers=observed_numbers,
        )

    # --- SUY LUáº¬N: OCR khÃ´ng dá»±ng Ä‘Æ°á»£c dÃ£y 1,2,â€¦ há»£p lá»‡, nhÆ°ng cÃ³ nhÃ£n "CÃCH"
    # (any_cach) nghÄ©a lÃ  video CHáº®C CHáº®N nhiá»u cÃ¡ch. DÃ¹ng rÃ ng buá»™c "sá»‘ cÃ¡ch lÃ 
    # dÃ£y liÃªn tiáº¿p 1,2,3,â€¦" + cÃ¡c khá»‘i giÃ¡ trá»‹ á»•n Ä‘á»‹nh theo thá»i gian Ä‘á»ƒ khÃ´i
    # phá»¥c ranh giá»›i. Náº¿u suy luáº­n Ä‘Æ°á»£c â†’ multi (inferred=True, vÃ o inferred_review
    # Ä‘á»ƒ ngÆ°á»i kiá»ƒm láº¡i). Náº¿u khÃ´ng â†’ manual_review (cáº¯t tay). ---
    if any_cach:
        inferred = _infer_multi_from_cach(samples, lo, hi, confirm, video_id)
        if inferred is not None:
            return inferred

    # Req 4.3/4.7: má»i trÆ°á»ng há»£p báº¥t thÆ°á»ng cÃ²n láº¡i â†’ cáº§n rÃ  soÃ¡t thá»§ cÃ´ng.
    return SegmentationResult(
        video_id=video_id,
        kind="manual_review",
        variant_count=0,
        spans=(),
        observed_numbers=observed_numbers,
    )


def _stable_blocks(
    samples: list[Sample], confirm: int
) -> list[tuple[int, int, int]]:
    """RÃºt cÃ¡c KHá»I giÃ¡ trá»‹ sá»‘ á»”N Äá»ŠNH theo thá»i gian (lá»c nhiá»…u láº» + None).

    Má»™t "khá»‘i" lÃ  má»™t Ä‘oáº¡n cÃ¡c frame máº«u **liÃªn tiáº¿p** cÃ¹ng mang má»™t giÃ¡ trá»‹ sá»‘
    ``v`` (bá» qua ``None`` xen giá»¯a: None = khÃ´ng quan sÃ¡t, khÃ´ng phÃ¡ vá»¡ khá»‘i) vá»›i
    Ä‘á»™ dÃ i (sá»‘ frame máº«u thá»±c sá»± mang ``v``) ``>= confirm``. GiÃ¡ trá»‹ chá»‰ thoÃ¡ng
    qua < ``confirm`` (vd "2" láº» 1 frame cá»§a W00738, hay "7" nhiá»…u) bá»‹ loáº¡i.

    Tráº£ danh sÃ¡ch ``(value, start_frame, end_frame)`` theo thá»© tá»± thá»i gian, vá»›i
    ``start_frame``/``end_frame`` lÃ  frame máº«u Ä‘áº§u/cuá»‘i cá»§a khá»‘i (Ä‘Ã£ gá»™p Ä‘uÃ´i
    None vÃ o khá»‘i liá»n trÆ°á»›c Ä‘á»ƒ phá»§ kÃ­n thá»i gian).
    """
    # Gom run liÃªn tiáº¿p cÃ¹ng giÃ¡ trá»‹ (None lÃ  má»™t "giÃ¡ trá»‹" riÃªng táº¡m thá»i).
    runs: list[tuple[int | None, int, int, int]] = []  # (val, count, f0, f1)
    for frame_index, num in samples:
        if runs and runs[-1][0] == num:
            v, c, f0, _ = runs[-1]
            runs[-1] = (v, c + 1, f0, frame_index)
        else:
            runs.append((num, 1, frame_index, frame_index))

    # Giá»¯ cÃ¡c run sá»‘ (khÃ´ng None) Ä‘á»§ dÃ i >= confirm; Ä‘Ã¢y lÃ  cÃ¡c khá»‘i á»•n Ä‘á»‹nh.
    blocks: list[tuple[int, int, int]] = []
    for val, count, f0, f1 in runs:
        if val is None:
            continue
        if count >= confirm:
            blocks.append((val, f0, f1))

    # Há»£p nháº¥t cÃ¡c khá»‘i liá»n nhau cÃ¹ng giÃ¡ trá»‹ (phÃ²ng khi None chen giá»¯a tÃ¡ch Ä‘Ã´i).
    merged: list[tuple[int, int, int]] = []
    for val, f0, f1 in blocks:
        if merged and merged[-1][0] == val:
            pv, pf0, _ = merged[-1]
            merged[-1] = (pv, pf0, f1)
        else:
            merged.append((val, f0, f1))
    return merged


def _infer_multi_from_cach(
    samples: list[Sample],
    lo: int,
    hi: int,
    confirm: int,
    video_id: str,
) -> "SegmentationResult | None":
    """SUY LUáº¬N dÃ£y cÃ¡ch 1,2,â€¦,N tá»« cÃ¡c khá»‘i á»•n Ä‘á»‹nh khi OCR Ä‘á»c thiáº¿u/nhiá»…u.

    Tiá»n Ä‘á»: Ä‘Ã£ biáº¿t video cÃ³ nhÃ£n "CÃCH" (``any_cach``) nÃªn CHáº®C CHáº®N nhiá»u cÃ¡ch.
    RÃ ng buá»™c miá»n: sá»‘ cÃ¡ch lÃ  dÃ£y nguyÃªn LIÃŠN TIáº¾P báº¯t Ä‘áº§u tá»« 1 theo thá»i gian.

    Chiáº¿n lÆ°á»£c (chá»‰ tÃ¡ch tá»± Ä‘á»™ng khi cÃ³ ÃT NHáº¤T Má»˜T Ä‘iá»ƒm chuyá»ƒn rÃµ â€” tá»©c >= 2 khá»‘i
    á»•n Ä‘á»‹nh khÃ¡c giÃ¡ trá»‹):

    1. Láº¥y cÃ¡c khá»‘i á»•n Ä‘á»‹nh (:func:`_stable_blocks`). Cáº§n ``>= 2`` khá»‘i **khÃ¡c
       giÃ¡ trá»‹ nhau** liá»n ká» (má»™t Ä‘iá»ƒm chuyá»ƒn thá»±c sá»±). Chá»‰ má»™t khá»‘i (vd chá»‰ "2"
       nhÆ° D0105 mÃ  "1" khÃ´ng Ä‘á»c Ä‘Æ°á»£c) váº«n coi lÃ  má»™t Ä‘iá»ƒm chuyá»ƒn: Ä‘oáº¡n trÆ°á»›c
       khá»‘i "2" chÃ­nh lÃ  cÃ¡ch 1 (suy ngÆ°á»£c).
    2. Bá» cÃ¡c giÃ¡ trá»‹ trÃ¹ng láº·p liÃªn tiáº¿p; sá»‘ khá»‘i phÃ¢n biá»‡t = sá»‘ "má»‘c" quan sÃ¡t.
       KhÃ´i phá»¥c nhÃ£n cÃ¡ch theo thá»© tá»± thá»i gian thÃ nh 1,2,3,â€¦ Báº¤T Ká»‚ OCR Ä‘á»c ra
       sá»‘ gÃ¬ (OCR cÃ³ thá»ƒ nháº§m 2â†’7); chá»‰ dÃ¹ng Vá»Š TRÃ chuyá»ƒn, khÃ´ng tin nhÃ£n sá»‘.
    3. Ranh giá»›i Ä‘áº·t táº¡i Ä‘áº§u má»—i khá»‘i (trá»« khá»‘i Ä‘áº§u báº¯t Ä‘áº§u tá»« ``lo``); dá»±ng span
       liÃªn tá»¥c phá»§ ``[lo, hi]``.

    Tráº£ :class:`SegmentationResult` ``kind="multi", inferred=True`` náº¿u suy luáº­n
    Ä‘Æ°á»£c (>= 2 cÃ¡ch); ``None`` náº¿u khÃ´ng Ä‘á»§ tÃ­n hiá»‡u (Ä‘á»ƒ phÃ­a gá»i â†’ manual_review).
    """
    observed = tuple(num for _, num in samples)
    blocks = _stable_blocks(samples, confirm)
    if not blocks:
        return None

    # Chá»‰ suy luáº­n khi cÃ³ >= 2 khá»‘i á»•n Ä‘á»‹nh khÃ¡c giÃ¡ trá»‹ â€” tá»©c cÃ³ Ã­t nháº¥t 1 Ä‘iá»ƒm
    # chuyá»ƒn thá»±c sá»± quan sÃ¡t Ä‘Æ°á»£c. KhÃ´ng suy ngÆ°á»£c tá»« Ä‘oáº¡n None Ä‘áº§u video: Ä‘oáº¡n
    # None trÆ°á»›c khá»‘i Ä‘áº§u cÃ³ thá»ƒ chá»‰ lÃ  overlay chÆ°a rÃµ rÃ ng, khÃ´ng pháº£i cÃ¡ch riÃªng.
    if len(blocks) < 2:
        return None

    first_val, first_f0, _ = blocks[0]

    # Ranh giá»›i Ä‘áº·t táº¡i frame Ä‘áº§u cá»§a má»—i khá»‘i á»•n Ä‘á»‹nh; khá»‘i Ä‘áº§u báº¯t Ä‘áº§u tá»« lo.
    starts: list[int] = [lo]
    for _, f0, _ in blocks[1:]:
        starts.append(f0)

    starts = sorted(set(starts))
    n_variants = len(starts)

    # Chá»‰ tÃ¡ch tá»± Ä‘á»™ng khi cÃ³ Ä‘iá»ƒm chuyá»ƒn thá»±c sá»± (>= 2 cÃ¡ch).
    if n_variants < 2:
        return None

    spans: list[VariantSpan] = []
    for k in range(n_variants):
        start = starts[k]
        end = hi if k == n_variants - 1 else starts[k + 1] - 1
        if end < start:
            return None  # má»‘c chá»“ng láº¥n báº¥t thÆ°á»ng â†’ Ä‘á»ƒ manual_review
        spans.append(
            VariantSpan(variant_index=k + 1, start_frame=start, end_frame=end)
        )

    return SegmentationResult(
        video_id=video_id,
        kind="multi",
        variant_count=n_variants,
        spans=tuple(spans),
        observed_numbers=observed,
        inferred=True,
    )


# Má»™t "nguá»“n frame" lÃ  callable nháº­n chá»‰_sá»‘_frame (0-based) vÃ  tráº£ vá» frame
# (máº£ng numpy) hoáº·c ``None`` náº¿u khÃ´ng Ä‘á»c Ä‘Æ°á»£c frame Ä‘Ã³.
FrameReader = Callable[[int], "object | None"]


def _make_frame_reader(video) -> "FrameReader":
    """Chuáº©n hÃ³a *video* thÃ nh má»™t callable ``read(frame_index) -> frame | None``.

    Cho phÃ©p :func:`refine_boundary` Ä‘á»c má»™t frame báº¥t ká»³ theo chá»‰ sá»‘ mÃ  khÃ´ng
    phá»¥ thuá»™c cá»©ng vÃ o OpenCV â€” thuáº­n tiá»‡n cho property-test (Property 7) chá»‰ cáº§n
    truyá»n má»™t hÃ m thuáº§n Ã¡nh xáº¡ ``frame_index -> frame``.

    *video* cÃ³ thá»ƒ lÃ :

    * má»™t **callable** ``frame_index -> frame | None`` (dÃ¹ng nguyÃªn); hoáº·c
    * má»™t Ä‘á»‘i tÆ°á»£ng kiá»ƒu :class:`cv2.VideoCapture` (cÃ³ ``set`` + ``read``): khi Ä‘Ã³
      Ä‘á»c frame báº±ng ``set(CAP_PROP_POS_FRAMES, index)`` rá»“i ``read()``.

    Args:
        video: Nguá»“n frame (callable hoáº·c capture giá»‘ng ``cv2.VideoCapture``).

    Returns:
        Callable Ä‘á»c frame theo chá»‰ sá»‘; tráº£ ``None`` khi khÃ´ng Ä‘á»c Ä‘Æ°á»£c.
    """
    if callable(video):
        return video

    # Giáº£ Ä‘á»‹nh Ä‘á»‘i tÆ°á»£ng kiá»ƒu cv2.VideoCapture.
    import cv2  # import trá»…: chá»‰ cáº§n khi thá»±c sá»± Ä‘á»c tá»« capture

    def _read(index: int):
        try:
            video.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            grabbed, frame = video.read()
        except Exception:  # noqa: BLE001 - capture lá»—i -> coi nhÆ° khÃ´ng Ä‘á»c Ä‘Æ°á»£c
            return None
        if not grabbed or frame is None:
            return None
        return frame

    return _read


def refine_boundary(
    detector: "NumberDetector",
    video,
    coarse_lo: int,
    coarse_hi: int,
    target_number: int,
    cfg: "PreprocessConfig",
    logger: "logging.Logger | None" = None,
) -> int:
    """Tinh chá»‰nh má»™t ranh giá»›i thÃ´ vá» frame Äáº¦U TIÃŠN mang ``target_number``.

    Hiá»‡n thá»±c Req 4.4: cho má»™t ranh giá»›i thÃ´ náº±m giá»¯a hai frame máº«u
    ``coarse_lo`` (frame máº«u mang giÃ¡ trá»‹ **cÅ©**) vÃ  ``coarse_hi`` (frame máº«u Ä‘áº§u
    tiÃªn mang giÃ¡ trá»‹ **má»›i** ``target_number``), tÃ¬m chá»‰ sá»‘ frame ``r`` Ä‘áº§u tiÃªn
    trong khoáº£ng ``(coarse_lo, coarse_hi]`` mÃ  :meth:`NumberDetector.detect` Ä‘á»c
    Ä‘Æ°á»£c Ä‘Ãºng ``target_number``. VÃ¬ detector tá»•ng há»£p lÃ  **Ä‘Æ¡n Ä‘iá»‡u** (giÃ¡ trá»‹ cÅ©
    trÆ°á»›c Ä‘iá»ƒm chuyá»ƒn ``T``, giÃ¡ trá»‹ má»›i tá»« ``T`` trá»Ÿ Ä‘i), tÃ¬m-nhá»‹-phÃ¢n cho ra
    Ä‘Ãºng ``T``; trong má»i trÆ°á»ng há»£p sai sá»‘ ``|r âˆ’ T| â‰¤ 1`` (Property 7).

    Xá»­ lÃ½ lá»—i (Req 4.8): náº¿u Ä‘á»c frame tháº¥t báº¡i, hoáº·c ``detector.detect`` nÃ©m lá»—i
    / quÃ¡ thá»i gian trÃªn **má»™t** frame, frame Ä‘Ã³ Ä‘Æ°á»£c **coi nhÆ° khÃ´ng cÃ³ sá»‘**
    (khÃ´ng mang ``target_number``), ghi log, rá»“i tiáº¿p tá»¥c â€” khÃ´ng lÃ m há»ng cáº£ quÃ¡
    trÃ¬nh tinh chá»‰nh.

    Args:
        detector: Bá»™ phÃ¡t hiá»‡n sá»‘ (:class:`NumberDetector`).
        video: Nguá»“n frame â€” callable ``frame_index -> frame`` hoáº·c capture kiá»ƒu
            ``cv2.VideoCapture`` (xem :func:`_make_frame_reader`).
        coarse_lo: Chá»‰ sá»‘ frame máº«u mang giÃ¡ trá»‹ cÅ© (cáº­n dÆ°á»›i, **loáº¡i trá»«**).
        coarse_hi: Chá»‰ sá»‘ frame máº«u Ä‘áº§u tiÃªn mang ``target_number`` (cáº­n trÃªn,
            **bao gá»“m**) â€” cÅ©ng lÃ  giÃ¡ trá»‹ máº·c Ä‘á»‹nh khi khÃ´ng tÃ¬m tháº¥y frame nÃ o
            khÃ¡c mang sá»‘ má»›i.
        target_number: GiÃ¡ trá»‹ sá»‘ má»›i cáº§n Ä‘á»‹nh vá»‹ Ä‘iá»ƒm xuáº¥t hiá»‡n Ä‘áº§u tiÃªn.
        cfg: :class:`~qipedc_video_preprocess.config.PreprocessConfig` (giá»¯ trong
            chá»¯ kÃ½ Ä‘á»ƒ Ä‘á»“ng nháº¥t interface; hiá»‡n chÆ°a dÃ¹ng tham sá»‘ nÃ o trá»±c tiáº¿p).
        logger: Logger tÃ¹y chá»n cho lá»—i frame (Req 4.8); máº·c Ä‘á»‹nh logger module.

    Returns:
        Chá»‰ sá»‘ frame ``r`` (0-based) Ä‘áº§u tiÃªn mang ``target_number``, náº±m trong
        ``(coarse_lo, coarse_hi]``; tráº£ vá» ``coarse_hi`` náº¿u khÃ´ng frame nÃ o trong
        khoáº£ng Ä‘Æ°á»£c xÃ¡c nháº­n mang sá»‘ má»›i.
    """
    log = logger if logger is not None else globals()["logger"]
    read = _make_frame_reader(video)

    lo = int(coarse_lo)
    hi = int(coarse_hi)
    # Khoáº£ng tÃ¬m kiáº¿m rá»—ng/Ä‘áº£o -> khÃ´ng cÃ³ gÃ¬ Ä‘á»ƒ tinh chá»‰nh, tráº£ cáº­n trÃªn.
    if hi <= lo:
        return hi

    def _bears_target(index: int) -> bool:
        """True náº¿u frame *index* Ä‘Æ°á»£c detector Ä‘á»c ra Ä‘Ãºng ``target_number``."""
        frame = read(index)
        if frame is None:
            log.warning(
                "refine_boundary: khÃ´ng Ä‘á»c Ä‘Æ°á»£c frame %d -> coi nhÆ° khÃ´ng cÃ³ sá»‘ "
                "(Req 4.8).",
                index,
            )
            return False
        try:
            result = detector.detect(frame)
        except Exception as exc:  # noqa: BLE001 - Req 4.8: lá»—i/timeout 1 frame
            log.warning(
                "refine_boundary: detector lá»—i táº¡i frame %d (%s: %s) -> coi nhÆ° "
                "khÃ´ng cÃ³ sá»‘, tiáº¿p tá»¥c (Req 4.8).",
                index,
                type(exc).__name__,
                exc,
            )
            return False
        return result.number == target_number

    # TÃ¬m-nhá»‹-phÃ¢n cáº­n-dÆ°á»›i (lower bound) chá»‰ sá»‘ Ä‘áº§u tiÃªn mang target_number trong
    # (coarse_lo, coarse_hi]. coarse_hi Ä‘Ã£ biáº¿t mang sá»‘ má»›i (frame máº«u) nÃªn luÃ´n lÃ 
    # á»©ng viÃªn há»£p lá»‡ máº·c Ä‘á»‹nh.
    search_lo = lo + 1
    search_hi = hi
    answer = hi
    while search_lo <= search_hi:
        mid = (search_lo + search_hi) // 2
        if _bears_target(mid):
            answer = mid
            search_hi = mid - 1
        else:
            search_lo = mid + 1

    return answer


def detect_overlay_change_boundary(
    video,
    roi: tuple[float, float, float, float],
    fps: float,
    num_frames: int,
    cfg: "PreprocessConfig",
    logger: "logging.Logger | None" = None,
) -> int | None:
    """Äá»‹nh vá»‹ Ä‘iá»ƒm overlay "cÃ¡ch" Äá»”I (cÃ¡ch 1â†’2) báº±ng chÃªnh lá»‡ch pixel vÃ¹ng ROI.

    DÃ¹ng cho SUY LUáº¬N khi OCR khÃ´ng Ä‘á»c ná»•i con sá»‘ (bug W00738): logo che gáº§n kÃ­n
    chá»¯ "CÃCH 1/2" nhÆ°ng khi nhÃ£n Ä‘á»•i 1â†’2 váº«n cÃ³ má»™t thay Ä‘á»•i pixel nhá» trong vÃ¹ng
    overlay gÃ³c trÃªn trÃ¡i. HÃ m láº¥y máº«u dÃ y vÃ¹ng ROI, tÃ­nh mean ``cv2.absdiff`` giá»¯a
    cÃ¡c frame máº«u liÃªn tiáº¿p, vÃ  tráº£ vá» frame nÆ¡i chÃªnh lá»‡ch Ä‘áº¡t Äá»ˆNH rÃµ rá»‡t (vÆ°á»£t
    bá»™i sá»‘ ``multiview_diff``-style so vá»›i ná»n), miá»…n lÃ  Ä‘á»‰nh náº±m cÃ¡ch hai biÃªn Ã­t
    nháº¥t ``min_gap``. Tráº£ ``None`` náº¿u khÃ´ng cÃ³ Ä‘á»‰nh Ä‘á»§ rÃµ (overlay tÄ©nh / nhiá»…u).

    ÄÃ¢y lÃ  logic biÃªn (Ä‘á»c frame + cv2); ranh giá»›i tráº£ vá» lÃ  Ä‘iá»ƒm báº¯t Ä‘áº§u cÃ¡ch káº¿
    tiáº¿p (0-based). Video tÃ¡ch theo Ä‘iá»ƒm nÃ y Ä‘Æ°á»£c Ä‘Ã¡nh ``inferred=True`` â†’ ngÆ°á»i
    kiá»ƒm láº¡i trong ``split_variant_predicted``.
    """
    log = logger if logger is not None else globals()["logger"]
    if num_frames <= 1 or fps <= 0:
        return None

    import cv2  # import trá»…
    import numpy as np

    from .number_detector import crop_roi

    read = _make_frame_reader(video)

    # Láº¥y máº«u dÃ y hÆ¡n bÆ°á»›c phÃ¢n loáº¡i Ä‘á»ƒ Ä‘á»‹nh vá»‹ Ä‘iá»ƒm Ä‘á»•i sáº¯c nÃ©t hÆ¡n (~3 máº«u/giÃ¢y).
    step = max(1, round(float(fps) * 0.33))
    indices = list(range(0, int(num_frames), step))
    if len(indices) < 3:
        return None

    crops: list = []
    valid_idx: list[int] = []
    for idx in indices:
        frame = read(idx)
        if frame is None:
            continue
        roi_img = crop_roi(frame, roi)
        if roi_img is None or getattr(roi_img, "size", 0) == 0:
            continue
        try:
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        except Exception:  # noqa: BLE001 - frame lá»—i â†’ bá» qua máº«u nÃ y
            continue
        crops.append(gray)
        valid_idx.append(idx)

    if len(crops) < 3:
        return None

    # ChÃªnh lá»‡ch trung bÃ¬nh giá»¯a hai máº«u liÃªn tiáº¿p, chuáº©n hÃ³a [0,1].
    diffs: list[float] = []
    for a, b in zip(crops, crops[1:]):
        if a.shape != b.shape:
            h = min(a.shape[0], b.shape[0])
            w = min(a.shape[1], b.shape[1])
            a2, b2 = a[:h, :w], b[:h, :w]
        else:
            a2, b2 = a, b
        diffs.append(float(np.mean(cv2.absdiff(a2, b2))) / 255.0)

    if not diffs:
        return None

    # Loáº¡i cÃ¡c Ä‘iá»ƒm quÃ¡ gáº§n hai biÃªn (min_gap giÃ¢y) Ä‘á»ƒ trÃ¡nh nhiá»…u má»Ÿ/Ä‘Ã³ng overlay.
    min_gap = max(1, round(float(fps) * float(cfg.multiview_min_gap_seconds)))
    peak_pos = -1
    peak_val = 0.0
    for i, d in enumerate(diffs):
        boundary_frame = valid_idx[i + 1]
        if boundary_frame < min_gap or boundary_frame > (num_frames - 1 - min_gap):
            continue
        if d > peak_val:
            peak_val = d
            peak_pos = i

    if peak_pos < 0:
        return None

    # Ná»n = trung vá»‹ cÃ¡c chÃªnh lá»‡ch cÃ²n láº¡i; Ä‘á»‰nh pháº£i ná»•i báº­t rÃµ so vá»›i ná»n.
    others = [d for j, d in enumerate(diffs) if j != peak_pos]
    baseline = float(np.median(others)) if others else 0.0
    if peak_val < float(cfg.multiview_diff_threshold) and peak_val < 4.0 * (baseline + 1e-6):
        log.info(
            "detect_overlay_change_boundary: Ä‘á»‰nh chÃªnh lá»‡ch %.4f khÃ´ng Ä‘á»§ rÃµ so "
            "vá»›i ná»n %.4f â†’ khÃ´ng suy luáº­n Ä‘Æ°á»£c Ä‘iá»ƒm Ä‘á»•i cÃ¡ch.",
            peak_val,
            baseline,
        )
        return None

    boundary = valid_idx[peak_pos + 1]
    log.info(
        "detect_overlay_change_boundary: Ä‘iá»ƒm Ä‘á»•i overlay táº¡i frame %d (diff=%.4f, "
        "ná»n=%.4f).",
        boundary,
        peak_val,
        baseline,
    )
    return boundary


def segment_video(
    entry: "VideoEntry",
    props: "VideoProps",
    detector: "NumberDetector",
    cfg: "PreprocessConfig",
    logger: "logging.Logger | None" = None,
) -> SegmentationResult:
    """PhÃ¢n Ä‘oáº¡n má»™t video: láº¥y máº«u â†’ phÃ¡t hiá»‡n â†’ phÃ¢n loáº¡i â†’ tinh chá»‰nh ranh giá»›i.

    Äiá»u phá»‘i toÃ n bá»™ Requirement 4 cho má»™t video:

    1. Sinh chá»‰ sá»‘ frame máº«u cÃ¡ch Ä‘á»u báº±ng :func:`sample_frame_indices` theo
       ``props.fps`` / ``props.num_frames`` vÃ  ``cfg.sample_interval_seconds``
       (Req 4.1).
    2. Äá»c tá»«ng frame máº«u tá»« ``entry.path`` báº±ng :class:`cv2.VideoCapture` vÃ  gá»i
       ``detector.detect``; lá»—i/timeout cá»§a detector trÃªn má»™t frame â†’ coi frame Ä‘Ã³
       **khÃ´ng cÃ³ sá»‘**, ghi log, tiáº¿p tá»¥c (Req 4.8).
    3. PhÃ¢n loáº¡i chuá»—i ``(frame_index, number|None)`` báº±ng :func:`classify_sequence`.
    4. Vá»›i video nhiá»u cÃ¡ch, tinh chá»‰nh tá»«ng ranh giá»›i thÃ´ vá» sai sá»‘ â‰¤ 1 frame
       báº±ng :func:`refine_boundary` (Req 4.4), rá»“i dá»±ng láº¡i cÃ¡c :class:`VariantSpan`
       liÃªn tá»¥c khÃ´ng chá»“ng láº¥n quanh cÃ¡c ranh giá»›i Ä‘Ã£ tinh chá»‰nh.

    Args:
        entry: :class:`~qipedc_video_preprocess.discovery.VideoEntry` cá»§a video.
        props: :class:`~qipedc_video_preprocess.video_probe.VideoProps` (fps, sá»‘
            frame) Ä‘Ã£ probe Ä‘Æ°á»£c.
        detector: Bá»™ phÃ¡t hiá»‡n sá»‘ (:class:`NumberDetector`).
        cfg: :class:`~qipedc_video_preprocess.config.PreprocessConfig`.
        logger: Logger tÃ¹y chá»n; máº·c Ä‘á»‹nh logger cá»§a module.

    Returns:
        :class:`SegmentationResult` vá»›i ``video_id`` cá»§a ``entry`` vÃ  cÃ¡c span Ä‘Ã£
        tinh chá»‰nh (vá»›i video nhiá»u cÃ¡ch).
    """
    log = logger if logger is not None else globals()["logger"]

    sample_indices = sample_frame_indices(
        props.fps, props.num_frames, cfg.sample_interval_seconds
    )

    import cv2  # import trá»…: pháº§n thuáº§n (classify/refine vá»›i callable) khÃ´ng cáº§n

    capture = cv2.VideoCapture(str(entry.path))
    try:
        if not capture.isOpened():
            log.warning(
                "segment_video: khÃ´ng má»Ÿ Ä‘Æ°á»£c video %s (video_id=%s) -> coi nhÆ° "
                "khÃ´ng cÃ³ frame máº«u nÃ o.",
                entry.path,
                entry.video_id,
            )

        samples: list[Sample] = []
        saw_cach_flags: list[bool] = []
        for index in sample_indices:
            number: int | None = None
            saw_cach = False
            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                grabbed, frame = capture.read()
            except Exception as exc:  # noqa: BLE001 - lá»—i Ä‘á»c frame
                log.warning(
                    "segment_video[%s]: lá»—i Ä‘á»c frame %d (%s: %s) -> coi nhÆ° "
                    "khÃ´ng cÃ³ sá»‘, tiáº¿p tá»¥c (Req 4.8).",
                    entry.video_id,
                    index,
                    type(exc).__name__,
                    exc,
                )
                grabbed, frame = False, None

            if grabbed and frame is not None:
                try:
                    detection = detector.detect(frame)
                    number = detection.number
                    saw_cach = getattr(detection, "saw_cach", False)
                except Exception as exc:  # noqa: BLE001 - Req 4.8
                    log.warning(
                        "segment_video[%s]: detector lá»—i táº¡i frame %d (%s: %s) -> "
                        "coi nhÆ° khÃ´ng cÃ³ sá»‘, tiáº¿p tá»¥c (Req 4.8).",
                        entry.video_id,
                        index,
                        type(exc).__name__,
                        exc,
                    )
                    number = None
            samples.append((index, number))
            saw_cach_flags.append(saw_cach)

        result = classify_sequence(
            samples, cfg, video_id=entry.video_id, saw_cach_flags=saw_cach_flags
        )

        boundary_method = getattr(cfg, "boundary_method", "ocr")
        suspected_multi_variant = result.kind != "single" or any(saw_cach_flags)
        if (
            boundary_method in {"pose", "ensemble", "calibrated"}
            and suspected_multi_variant
        ):
            selected = _segment_with_pose_or_ensemble(
                entry=entry,
                props=props,
                cfg=cfg,
                capture=capture,
                detector=detector,
                sample_indices=sample_indices,
                result=result,
                saw_cach_flags=saw_cach_flags,
                logger_obj=log,
                boundary_method=boundary_method,
            )
            if selected is not None:
                return selected

        # --- SUY LUáº¬N báº±ng THAY Äá»”I OVERLAY (bug W00738) ---
        # classify tráº£ manual_review nhÆ°ng cÃ³ tÃ­n hiá»‡u "cÃ¡ch" (ká»ƒ cáº£ máº£nh yáº¿u): thá»­
        # tÃ¬m Ä‘iá»ƒm overlay Ä‘á»•i cÃ¡ch báº±ng frame-diff vÃ¹ng overlay. CÃ³ Ä‘iá»ƒm Ä‘á»§ rÃµ â†’
        # multi inferred (split_variant_predicted); khÃ´ng â†’ giá»¯ manual (cáº¯t tay).
        if result.kind == "manual_review" and any(saw_cach_flags):
            boundary = detect_overlay_change_boundary(
                capture,
                cfg.roi_top_left,
                props.fps,
                props.num_frames,
                cfg,
                logger=log,
            )
            if boundary is not None:
                lo = sample_indices[0] if sample_indices else 0
                hi = props.num_frames - 1
                min_frames = max(
                    1, round(float(cfg.min_variant_seconds) * float(props.fps))
                )
                cand = (
                    VariantSpan(1, lo, boundary - 1),
                    VariantSpan(2, boundary, hi),
                )
                if all(
                    s.end_frame - s.start_frame + 1 >= min_frames for s in cand
                ):
                    log.info(
                        "segment_video[%s]: SUY LUáº¬N multi tá»« thay Ä‘á»•i overlay táº¡i "
                        "frame %d â†’ split_variant_predicted.",
                        entry.video_id,
                        boundary,
                    )
                    return SegmentationResult(
                        video_id=entry.video_id,
                        kind="multi",
                        variant_count=2,
                        spans=cand,
                        observed_numbers=result.observed_numbers,
                        inferred=True,
                    )
            # KhÃ´ng tÃ¬m Ä‘Æ°á»£c Ä‘iá»ƒm Ä‘á»•i Ä‘á»§ rÃµ â†’ giá»¯ manual_review Ä‘á»ƒ cáº¯t tay.
            return result

        # Chá»‰ video nhiá»u cÃ¡ch má»›i cÃ³ ranh giá»›i ná»™i bá»™ cáº§n tinh chá»‰nh (Req 4.4).
        if result.kind != "multi" or len(result.spans) < 2:
            return result

        # Video SUY LUáº¬N: OCR khÃ´ng Ä‘á»c ná»•i con sá»‘ nÃªn refine_boundary (dÃ² theo sá»‘)
        # sáº½ khÃ´ng tÃ¬m tháº¥y gÃ¬ â†’ giá»¯ nguyÃªn ranh giá»›i thÃ´ tá»« cÃ¡c khá»‘i á»•n Ä‘á»‹nh. CÃ¡c
        # video nÃ y sáº½ Ä‘Æ°á»£c ngÆ°á»i kiá»ƒm láº¡i trong inferred_review.
        if result.inferred:
            log.info(
                "segment_video[%s]: dÃ¹ng ranh giá»›i SUY LUáº¬N (inferred) â€” bá» qua "
                "refine theo sá»‘, gom vÃ o inferred_review.",
                entry.video_id,
            )
            return result

        coarse_spans = result.spans
        sample_frames = [idx for idx, _ in samples]
        n_variants = len(coarse_spans)

        # Vá»›i má»—i ranh giá»›i thÃ´ (giá»¯a cÃ¡ch k vÃ  k+1), giÃ¡ trá»‹ sá»‘ má»›i lÃ  k+1; frame
        # máº«u Ä‘áº§u tiÃªn mang giÃ¡ trá»‹ má»›i lÃ  start_frame cá»§a span k (0-based theo
        # thá»© tá»±). coarse_lo lÃ  frame máº«u liá»n trÆ°á»›c trong chuá»—i máº«u.
        refined: list[int] = []
        for k in range(1, n_variants):
            coarse_hi = coarse_spans[k].start_frame
            target_number = k + 1

            # frame máº«u liá»n trÆ°á»›c frame máº«u mang sá»‘ má»›i = cáº­n dÆ°á»›i (loáº¡i trá»«).
            try:
                pos = sample_frames.index(coarse_hi)
            except ValueError:
                pos = -1
            if pos > 0:
                coarse_lo = sample_frames[pos - 1]
            else:
                coarse_lo = max(0, coarse_hi - 1)

            r = refine_boundary(
                detector,
                capture,
                coarse_lo,
                coarse_hi,
                target_number,
                cfg,
                logger=log,
            )
            refined.append(r)

        # Dá»±ng láº¡i span liÃªn tá»¥c, khÃ´ng chá»“ng láº¥n quanh cÃ¡c ranh giá»›i Ä‘Ã£ tinh chá»‰nh.
        new_spans: list[VariantSpan] = []
        for k in range(n_variants):
            start = coarse_spans[0].start_frame if k == 0 else refined[k - 1]
            end = (
                coarse_spans[-1].end_frame
                if k == n_variants - 1
                else refined[k] - 1
            )
            new_spans.append(
                VariantSpan(variant_index=k + 1, start_frame=start, end_frame=end)
            )

        # --- Báº¢O Vá»† chá»‘ng ranh giá»›i tinh chá»‰nh sá»¥p Ä‘á»• (bug W00792) ---
        # Khi sá»‘ bá»‹ logo che gáº§n kÃ­n, detector Ä‘á»c ra "2" ngay tá»« frame ~1 nÃªn
        # refine_boundary sá»¥p ranh giá»›i vá» Ä‘áº§u video â†’ cÃ¡ch 1 chá»‰ cÃ²n [0,0], cáº£
        # video dá»“n vÃ o c2 (sai). Náº¿u Báº¤T Ká»² cÃ¡ch nÃ o sau tinh chá»‰nh ngáº¯n hÆ¡n
        # ngÆ°á»¡ng tá»‘i thiá»ƒu, ranh giá»›i OCR KHÃ”NG Ä‘Ã¡ng tin: KHÃ”NG phÃ¡t sinh clip suy
        # biáº¿n; quay vá» ranh giá»›i THÃ” (má»©c frame máº«u, vd frame 25) vÃ  Ä‘Ã¡nh dáº¥u
        # inferred=True Ä‘á»ƒ ngÆ°á»i kiá»ƒm láº¡i trong split_variant_predicted. Náº¿u ngay
        # cáº£ ranh giá»›i thÃ´ cÅ©ng cho cÃ¡ch quÃ¡ ngáº¯n â†’ manual_review (cáº¯t tay).
        min_variant_frames = max(
            1, round(float(cfg.min_variant_seconds) * float(props.fps))
        )

        def _shortest(spans: "tuple[VariantSpan, ...] | list[VariantSpan]") -> int:
            return min(s.end_frame - s.start_frame + 1 for s in spans)

        if _shortest(new_spans) < min_variant_frames:
            if _shortest(coarse_spans) < min_variant_frames:
                log.warning(
                    "segment_video[%s]: ranh giá»›i tinh chá»‰nh VÃ€ ranh giá»›i thÃ´ Ä‘á»u "
                    "cho cÃ¡ch < %d frame â€” OCR khÃ´ng Ä‘Ã¡ng tin â†’ manual_review.",
                    result.video_id,
                    min_variant_frames,
                )
                return SegmentationResult(
                    video_id=result.video_id,
                    kind="manual_review",
                    variant_count=0,
                    spans=(),
                    observed_numbers=result.observed_numbers,
                )
            log.warning(
                "segment_video[%s]: ranh giá»›i tinh chá»‰nh sá»¥p (cÃ¡ch < %d frame, bug "
                "W00792) â†’ dÃ¹ng ranh giá»›i THÃ” + inferred=True (split_variant_predicted).",
                result.video_id,
                min_variant_frames,
            )
            return SegmentationResult(
                video_id=result.video_id,
                kind="multi",
                variant_count=result.variant_count,
                spans=tuple(coarse_spans),
                observed_numbers=result.observed_numbers,
                inferred=True,
            )

        return SegmentationResult(
            video_id=result.video_id,
            kind=result.kind,
            variant_count=result.variant_count,
            spans=tuple(new_spans),
            observed_numbers=result.observed_numbers,
        )
    finally:
        capture.release()


def _segment_with_pose_or_ensemble(
    entry: "VideoEntry",
    props: "VideoProps",
    cfg: "PreprocessConfig",
    capture,
    detector: "NumberDetector",
    sample_indices: list[int],
    result: SegmentationResult,
    saw_cach_flags: list[bool],
    logger_obj: "logging.Logger",
    boundary_method: str,
) -> SegmentationResult | None:
    """Choose pose or ensemble boundaries without disturbing the OCR path."""
    log = logger_obj
    if boundary_method == "calibrated":
        from .boundary_calibration import predict_calibrated_boundaries

        calibrated = predict_calibrated_boundaries(
            entry.video_id,
            entry.path,
            props,
            cfg,
            expected_count=None,
            logger_obj=log,
        )
        selected = _result_from_boundaries(
            entry.video_id,
            props,
            list(calibrated.boundaries),
            result.observed_numbers,
            cfg,
            inferred=not calibrated.label_fitted,
        )
        if selected is not None:
            return selected
        return SegmentationResult(
            video_id=entry.video_id,
            kind="manual_review",
            variant_count=0,
            spans=(),
            observed_numbers=result.observed_numbers,
            inferred=True,
        )

    pose_boundaries = detect_method_boundaries(
        entry.path,
        props,
        cfg,
        start_frame=0,
        end_frame=props.num_frames - 1,
        logger_obj=log,
    )
    if boundary_method == "pose":
        selected = _result_from_boundaries(
            entry.video_id,
            props,
            pose_boundaries,
            result.observed_numbers,
            cfg,
            inferred=False,
        )
        if selected is not None:
            return selected
        return SegmentationResult(
            video_id=entry.video_id,
            kind="manual_review",
            variant_count=0,
            spans=(),
            observed_numbers=result.observed_numbers,
        )

    ocr_boundaries = _ocr_boundary_frames(
        result,
        capture,
        detector,
        sample_indices,
        cfg,
        props,
        saw_cach_flags,
        log,
    )
    if not pose_boundaries and not ocr_boundaries:
        return SegmentationResult(
            video_id=entry.video_id,
            kind="manual_review",
            variant_count=0,
            spans=(),
            observed_numbers=result.observed_numbers,
        )

    if pose_boundaries and ocr_boundaries:
        tolerance_frames = max(
            1, round(float(cfg.ensemble_tolerance_seconds) * float(props.fps))
        )
        sorted_pose = sorted(pose_boundaries)
        sorted_ocr = sorted(ocr_boundaries)
        chosen, matched_all_ocr = _align_ocr_boundaries_to_pose(
            sorted_ocr, sorted_pose, tolerance_frames
        )
        if matched_all_ocr:
            selected = _result_from_boundaries(
                entry.video_id,
                props,
                chosen,
                result.observed_numbers,
                cfg,
                inferred=False,
            )
            if selected is not None:
                return selected

        # OCR/overlay defines the method count; pose only refines/confirms it.
        # On videos that are both multi-method and multi-view, pose can expose
        # extra action/view boundaries. Those must not be promoted into extra
        # method variants (for example 2 methods x 2 views must stay c1/c2,
        # with view splitting handled later by _expand_views).
        selected = _result_from_boundaries(
            entry.video_id,
            props,
            chosen,
            result.observed_numbers,
            cfg,
            inferred=True,
        )
        if selected is not None:
            return selected

        # A collapsed OCR refinement (for example W00792 detecting "2" near
        # frame zero) is less credible than the same-count pose segmentation.
        # Preserve the method count, but use pose boundaries and route to review.
        if len(sorted_pose) == len(sorted_ocr):
            pose_fallback = _result_from_boundaries(
                entry.video_id,
                props,
                sorted_pose,
                result.observed_numbers,
                cfg,
                inferred=True,
            )
            if pose_fallback is not None:
                return pose_fallback

        return SegmentationResult(
            video_id=entry.video_id,
            kind="manual_review",
            variant_count=0,
            spans=(),
            observed_numbers=result.observed_numbers,
        )

    selected = _result_from_boundaries(
        entry.video_id,
        props,
        pose_boundaries or ocr_boundaries,
        result.observed_numbers,
        cfg,
        inferred=True,
    )
    if selected is not None:
        return selected

    return SegmentationResult(
        video_id=entry.video_id,
        kind="manual_review",
        variant_count=0,
        spans=(),
        observed_numbers=result.observed_numbers,
    )


def _ocr_boundary_frames(
    result: SegmentationResult,
    capture,
    detector: "NumberDetector",
    sample_indices: list[int],
    cfg: "PreprocessConfig",
    props: "VideoProps",
    saw_cach_flags: list[bool],
    logger_obj: "logging.Logger",
) -> list[int]:
    if result.kind == "multi" and result.spans:
        refined: list[int] = []
        for variant_index, span in enumerate(result.spans[1:], start=2):
            coarse_hi = int(span.start_frame)
            try:
                sample_pos = sample_indices.index(coarse_hi)
            except ValueError:
                sample_pos = -1
            if sample_pos > 0:
                coarse_lo = int(sample_indices[sample_pos - 1])
            else:
                sample_step = max(
                    1,
                    round(float(cfg.sample_interval_seconds) * float(props.fps)),
                )
                coarse_lo = max(0, coarse_hi - sample_step)
            refined.append(
                refine_boundary(
                    detector,
                    capture,
                    coarse_lo,
                    coarse_hi,
                    variant_index,
                    cfg,
                    logger=logger_obj,
                )
            )
        return refined

    if result.kind == "manual_review" and any(saw_cach_flags):
        boundary = detect_overlay_change_boundary(
            capture,
            cfg.roi_top_left,
            props.fps,
            props.num_frames,
            cfg,
            logger=logger_obj,
        )
        if boundary is not None:
            return [boundary]

    return []


def _align_ocr_boundaries_to_pose(
    ocr_boundaries: list[int],
    pose_boundaries: list[int],
    tolerance_frames: int,
) -> tuple[list[int], bool]:
    """Refine each OCR/method boundary with the nearest pose boundary.

    OCR/overlay owns the number of method variants. Pose is allowed to adjust an
    OCR boundary when there is a nearby action-rest transition, but surplus pose
    boundaries are ignored because they may be view/action boundaries within the
    same method.
    """
    if not ocr_boundaries:
        return [], False

    available = list(sorted(int(b) for b in pose_boundaries))
    chosen: list[int] = []
    matched_all = True
    for raw_ocr in sorted(int(b) for b in ocr_boundaries):
        best_index: int | None = None
        best_delta: int | None = None
        for idx, pose_boundary in enumerate(available):
            delta = abs(pose_boundary - raw_ocr)
            if best_delta is None or delta < best_delta:
                best_index = idx
                best_delta = delta
        if (
            best_index is not None
            and best_delta is not None
            and best_delta <= int(tolerance_frames)
        ):
            available.pop(best_index)
            # OCR/overlay is the first frame of the new method. Pose confirms
            # confidence only; averaging would move the cut into the new action.
            chosen.append(raw_ocr)
        else:
            chosen.append(raw_ocr)
            matched_all = False

    return chosen, matched_all


def _result_from_boundaries(
    video_id: str,
    props: "VideoProps",
    boundaries: list[int],
    observed_numbers: tuple[int | None, ...],
    cfg: "PreprocessConfig",
    *,
    inferred: bool,
) -> SegmentationResult | None:
    if not boundaries:
        return None

    lo = 0
    hi = max(0, int(props.num_frames) - 1)
    cuts = sorted({int(b) for b in boundaries if lo < int(b) <= hi})
    if not cuts:
        return None

    spans: list[VariantSpan] = []
    start = lo
    for idx, boundary in enumerate(cuts, start=1):
        end = boundary - 1
        if end < start:
            return None
        spans.append(VariantSpan(idx, start, end))
        start = boundary

    if start > hi:
        return None

    spans.append(VariantSpan(len(spans) + 1, start, hi))
    if any(span.end_frame < span.start_frame for span in spans):
        return None
    min_frames = max(1, round(float(cfg.min_variant_seconds) * float(props.fps)))
    if any(span.end_frame - span.start_frame + 1 < min_frames for span in spans):
        if inferred:
            return None
        inferred = True

    return SegmentationResult(
        video_id=video_id,
        kind="multi",
        variant_count=len(spans),
        spans=tuple(spans),
        observed_numbers=observed_numbers,
        inferred=inferred,
    )

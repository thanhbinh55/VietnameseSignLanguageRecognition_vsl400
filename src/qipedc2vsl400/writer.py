"""Emit the converted superset metadata as a VSL400-style view JSON (Req 5).

:func:`write_outputs` serializes a list of
:class:`~qipedc2vsl400.mapper.OutputRecord` objects to a JSON **array** at
``cfg.output_view_file`` (e.g. ``Dataset/final_dataset/front_view.json``).

The write is **deterministic** (Requirement 5.4 / design Property 4): records are
sorted ascending by ``video_id`` before serialization, so re-running the
converter on unchanged input yields byte-identical output. Vietnamese text is
preserved in UTF-8 with diacritics intact (``ensure_ascii=False``) rather than
ASCII-escaped to ``\\uXXXX`` (Requirement 5.2). The output directory is created
if it does not yet exist.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


def write_outputs(records: list[Any], cfg: Any, output_file: "Path | None" = None) -> None:
    """Write a superset metadata array to a view JSON file (deterministic).

    Args:
        records: The :class:`~qipedc2vsl400.mapper.OutputRecord` objects to
            emit. They are sorted ascending by ``video_id`` for deterministic
            output; the input list is not mutated.
        cfg: A :class:`~qipedc2vsl400.config.Config`-like object exposing
            ``output_view_file`` (the absolute ``front_view.json`` path).
        output_file: Optional explicit destination path; defaults to
            ``cfg.output_view_file`` (the front/main view). Lets the caller emit
            ``side_view.json`` with the same guarantees.

    Side effects:
        Creates the output directory if needed and writes a UTF-8 JSON file with
        ``indent=2`` and ``ensure_ascii=False`` (Vietnamese diacritics
        preserved). Re-running on unchanged ``records`` produces byte-identical
        output (Requirement 5.4).
    """
    target = output_file if output_file is not None else cfg.output_view_file
    target.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(records, key=lambda r: r.video_id)
    payload = [asdict(record) for record in ordered]

    with open(target, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _is_side_record(record: Any, cfg: Any) -> bool:
    """True if ``record.video_id`` carries the side-view suffix (e.g. ``_side``)."""
    suffix = getattr(cfg, "side_view_suffix", "_side")
    return str(record.video_id).endswith(suffix)


def write_view_outputs(records: list[Any], cfg: Any) -> tuple[int, int]:
    """Partition records by view and write ``front_view.json`` + ``side_view.json``.

    Multi-view clips are named ``<id>_side`` (or ``<id>_cK_side``) by the splitter;
    the ``cfg.side_view_suffix`` on ``video_id`` is the single routing signal. Front
    (main) records go to ``cfg.output_view_file`` and side records to
    ``cfg.side_view_file``. ``front_view.json`` is always (re)written; the
    ``side_view.json`` file is only written when at least one side record exists, so
    single-view runs keep producing exactly ``front_view.json`` as before.

    Returns:
        ``(num_front, num_side)`` record counts.
    """
    front = [r for r in records if not _is_side_record(r, cfg)]
    side = [r for r in records if _is_side_record(r, cfg)]

    write_outputs(front, cfg, output_file=cfg.output_view_file)
    if side:
        write_outputs(side, cfg, output_file=cfg.side_view_file)

    return len(front), len(side)

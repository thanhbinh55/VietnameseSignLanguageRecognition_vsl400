"""Tests cho phân tách front/side trong ``qipedc2vsl400.writer.write_view_outputs``."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from qipedc2vsl400.config import Config
from qipedc2vsl400.writer import write_view_outputs


@dataclass
class _Rec:
    video_id: str
    gloss: str


def _read(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def test_partitions_front_and_side():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(project_root=Path(tmp))
        records = [
            _Rec("W001", "a"),
            _Rec("W001_side", "a"),
            _Rec("W002_c2", "b"),
            _Rec("W002_c2_side", "b"),
        ]
        n_front, n_side = write_view_outputs(records, cfg)
        assert (n_front, n_side) == (2, 2)

        front = _read(cfg.output_view_file)
        side = _read(cfg.side_view_file)
        assert [r["video_id"] for r in front] == ["W001", "W002_c2"]
        assert [r["video_id"] for r in side] == ["W001_side", "W002_c2_side"]


def test_no_side_file_when_no_side_records():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(project_root=Path(tmp))
        records = [_Rec("W001", "a"), _Rec("W002", "b")]
        n_front, n_side = write_view_outputs(records, cfg)
        assert (n_front, n_side) == (2, 0)
        assert cfg.output_view_file.is_file()
        # side_view.json KHÔNG được tạo khi không có clip side (giữ hành vi cũ).
        assert not cfg.side_view_file.exists()


def test_front_output_is_deterministic_sorted():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(project_root=Path(tmp))
        records = [_Rec("W003", "c"), _Rec("W001", "a"), _Rec("W002", "b")]
        write_view_outputs(records, cfg)
        front = _read(cfg.output_view_file)
        assert [r["video_id"] for r in front] == ["W001", "W002", "W003"]

"""Đổi đường dẫn tuyệt đối trong Keypoint/processed/cam_*.json sang tương đối.

`keypoint/src/preprocess_dataset.py` (mã upstream) ghi các field ``video`` và
``pose`` bằng đường dẫn TUYỆT ĐỐI của máy đang chạy, ví dụ::

    "video": "D:\\...\\<Public>\\Processed\\metadata\\cam_front\\W03876_c1.mp4"
    "pose":  "D:\\...\\<Public>\\Keypoint\\processed\\cam_front\\W03876_c1_preprocessed.npy"

Khi phát hành dataset, đường dẫn đó vô nghĩa trên máy khác. Script này (bước hậu
kỳ — KHÔNG sửa mã upstream) đổi chúng thành đường dẫn **tương đối theo gốc folder
Public**, dùng dấu ``/`` cho cross-platform::

    "video": "Processed/metadata/cam_front/W03876_c1.mp4"
    "pose":  "Keypoint/processed/cam_front/W03876_c1_preprocessed.npy"

Cách cắt KHÔNG hard-code tiền tố máy: nó tìm neo cố định (``Keypoint/`` hoặc
``Processed/``) trong path rồi giữ từ neo đó trở đi — nên chạy đúng dù path gốc
khác máy/khác ổ đĩa.

Cách dùng::

    python scripts/relativize_keypoint_json.py \
        --public-root "D:/.../VSL Keypoint Dataset (QIPEDC-derived) - Public" \
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Các neo (anchor) cố định trong cây folder Public. Mọi path sau khi chuẩn hóa
# dấu phân tách đều phải chứa MỘT trong các neo này; phần từ neo trở đi là path
# tương đối cần giữ.
_ANCHORS = ("Keypoint/", "Processed/")
# Field chứa đường dẫn cần đổi.
_PATH_FIELDS = ("video", "pose")


def _to_relative(value: str) -> "str | None":
    """Trả về path tương đối (từ neo) nếu *value* là path chứa neo; ngược lại None."""
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("\\", "/")
    best: str | None = None
    for anchor in _ANCHORS:
        idx = normalized.rfind(anchor)
        if idx != -1:
            candidate = normalized[idx:]
            # Chọn neo cho path NGẮN nhất (gần gốc Public nhất).
            if best is None or len(candidate) < len(best):
                best = candidate
    return best


def relativize_file(path: Path, *, dry_run: bool) -> tuple[int, int]:
    """Đổi path trong một file JSON. Trả về (số record, số field đã đổi)."""
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} không phải JSON array")

    changed = 0
    for record in data:
        if not isinstance(record, dict):
            continue
        for field in _PATH_FIELDS:
            if field not in record:
                continue
            relative = _to_relative(record[field])
            if relative is not None and relative != record[field]:
                if not dry_run:
                    record[field] = relative
                changed += 1

    if dry_run:
        print(f"[dry-run] {path.name}: {len(data)} record, sẽ đổi {changed} field path")
    else:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=4)
        print(f"{path.name}: {len(data)} record, đã đổi {changed} field path → tương đối")
    return len(data), changed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--public-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    processed_dir = args.public_root.resolve() / "Keypoint" / "processed"
    targets = [processed_dir / "cam_front.json", processed_dir / "cam_side.json"]

    found = False
    for target in targets:
        if target.is_file():
            found = True
            relativize_file(target, dry_run=args.dry_run)
        else:
            print(f"Bỏ qua (không có): {target}", file=sys.stderr)

    if not found:
        print(
            f"Không tìm thấy cam_front.json/cam_side.json trong {processed_dir}.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

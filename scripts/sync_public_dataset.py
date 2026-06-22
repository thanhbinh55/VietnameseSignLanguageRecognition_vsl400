"""Đồng bộ output của pipeline (repo này) sang layout của thư mục dataset công bố
"VSL Keypoint Dataset (QIPEDC-derived) - Public".

Repo này có hai package tạo ra output:

* ``qipedc_video_preprocess`` — tách video → ``split_variants/*.mp4`` (clip một
  cách, một góc; ``_side`` = góc phụ) + ``labels_split.xlsx``.
* ``qipedc2vsl400`` — chia signer + convert metadata → ``metadata/front_view.json``
  (và ``side_view.json`` nếu có clip ``_side``) + ``signers.csv``.

Thư mục công bố dùng layout khác::

    Keypoint/{raw,processed}/{cam_front,cam_side}/   (.pose / *_preprocessed.npy)
    Processed/labels/labels.csv                      (video_id, gloss, gloss_id)
    Processed/metadata/cam_front.json                (= front_view.json + truy vết)
    Processed/metadata/cam_side.json                 (= side_view.json + truy vết)
    Processed/metadata/gloss.csv                     (gloss_id, gloss_text — không header)
    Processed/metadata/split_info.csv                (video_id, split)

Script này **không tự bịa** metadata: nó đọc ``front_view.json``/``side_view.json``
thật (đã có ``signer_id`` từ bước chia signer) và chỉ:

  1. Chuyển sang format ``cam_front.json``/``cam_side.json`` (đổi ``length_seconds``
     → ``length``, thêm truy vết ``parent_video_id`` / ``start_frame`` /
     ``end_frame``).
  2. Sinh ``labels.csv`` và ``gloss.csv`` (``gloss_id`` = vị trí của gloss trong
     danh sách gloss duy nhất đã sắp xếp — xác định, tái lập được).
  3. Sinh ``split_info.csv`` (train/val/test) bằng chia xác định theo
     ``parent_video_id`` (mọi clip cùng video gốc — ``_c1``/``_c2``/``_side`` —
     luôn cùng một split để tránh rò rỉ).
  4. (Tùy chọn ``--keypoints``) route ``.pose`` / ``*_preprocessed.npy`` đã có
     sẵn vào ``cam_front`` / ``cam_side`` theo hậu tố ``_side``.

Bản thân ``.pose``/``.npy`` được tạo ở **repo training**
(VietnameseSignLanguageRecognition: ``extract_keypoints.py`` →
``preprocess_dataset.py``), KHÔNG ở repo này — script chỉ định tuyến nếu chúng đã
nằm trong project.

Cách dùng::

    python scripts/sync_public_dataset.py \
        --project-root "D:/Duy/LVCF/DACD/split_convert" \
        --public-root  "D:/Duy/LVCF/DACD/split_convert/VSL Keypoint Dataset (QIPEDC-derived) - Public" \
        --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

# Các tỉ lệ split mặc định (xấp xỉ tỉ lệ bộ công bố 799/120/132 ≈ 0.76/0.115/0.125).
DEFAULT_SPLITS = (("train", 0.76), ("val", 0.115), ("test", 0.125))
_SIDE_SUFFIX = "_side"
_VARIANT_RE = re.compile(r"_c\d+$")


def _load_view_json(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else []


def _parent_video_id(video_id: str) -> str:
    """``D0024N_c1`` → ``D0024N``; ``W00144_c1_side`` → ``W00144``; ``D0093_side`` → ``D0093``.

    Bỏ hậu tố ``_side`` trước, rồi bỏ hậu tố cách ``_cN`` để về video QIPEDC gốc.
    """
    base = video_id[: -len(_SIDE_SUFFIX)] if video_id.endswith(_SIDE_SUFFIX) else video_id
    return _VARIANT_RE.sub("", base)


def _to_public_record(rec: dict) -> dict:
    """Đổi một record front_view/side_view → format cam_front/cam_side công bố.

    Giữ nguyên mọi field từ converter; đổi tên ``length_seconds`` → ``length`` và
    thêm truy vết ``parent_video_id`` / ``start_frame`` / ``end_frame``. Vì repo
    không lưu offset frame của clip con trong metadata, truy vết mặc định phủ toàn
    clip (``0 .. num_frames-1``); ``parent_video_id`` suy ra từ tên.
    """
    video_id = str(rec.get("video_id", ""))
    num_frames = int(rec.get("num_frames", 0) or 0)
    length = rec.get("length_seconds", rec.get("length", 0.0))
    return {
        "video_id": video_id,
        "signer_id": rec.get("signer_id", ""),
        "gloss": rec.get("gloss", ""),
        "fps": rec.get("fps", 0.0),
        "resolution": rec.get("resolution", 0),
        "length": length,
        "region": rec.get("region", ""),
        "topic": rec.get("topic", ""),
        "id": rec.get("id", ""),
        "num_frames": num_frames,
        "parent_video_id": _parent_video_id(video_id),
        "start_frame": 0,
        "end_frame": max(0, num_frames - 1),
    }


def _write_json(records: list[dict], dest: Path, dry_run: bool, label: str) -> None:
    if dry_run:
        print(f"[dry-run] {label}: {len(records)} record → {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: r["video_id"])
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(ordered, fh, ensure_ascii=False, indent=2)
    print(f"{label}: {len(records)} record → {dest}")


def _build_gloss_index(records: list[dict]) -> dict[str, int]:
    """``gloss_text`` → ``gloss_id`` = vị trí trong danh sách gloss duy nhất đã sắp xếp."""
    uniq = sorted({str(r.get("gloss", "")) for r in records})
    return {gloss: idx for idx, gloss in enumerate(uniq)}


def _write_labels_csv(records, gloss_index, dest: Path, dry_run: bool) -> None:
    rows = [
        (str(r["video_id"]), str(r.get("gloss", "")), gloss_index[str(r.get("gloss", ""))])
        for r in sorted(records, key=lambda r: r["video_id"])
    ]
    if dry_run:
        print(f"[dry-run] labels.csv: {len(rows)} dòng → {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["video_id", "gloss", "gloss_id"])
        writer.writerows(rows)
    print(f"labels.csv: {len(rows)} dòng → {dest}")


def _write_gloss_csv(gloss_index, dest: Path, dry_run: bool) -> None:
    rows = sorted(gloss_index.items(), key=lambda kv: kv[1])  # theo gloss_id
    if dry_run:
        print(f"[dry-run] gloss.csv: {len(rows)} gloss → {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        for gloss, gid in rows:
            writer.writerow([gid, gloss])  # không header
    print(f"gloss.csv: {len(rows)} gloss → {dest}")


def _assign_split(parent_id: str, seed: str, splits=DEFAULT_SPLITS) -> str:
    """Chia xác định theo ``parent_video_id`` để clip cùng video gốc cùng split.

    Hash ổn định (md5) → [0,1) → ánh xạ vào dải tỉ lệ tích lũy. Không phụ thuộc
    thứ tự duyệt, tái lập được giữa các lần chạy.
    """
    digest = hashlib.md5(f"{seed}:{parent_id}".encode("utf-8")).hexdigest()
    fraction = int(digest[:8], 16) / 0xFFFFFFFF
    cumulative = 0.0
    for name, ratio in splits:
        cumulative += ratio
        if fraction < cumulative:
            return name
    return splits[-1][0]


def _write_split_csv(records, dest: Path, seed: str, dry_run: bool) -> dict[str, int]:
    counts: dict[str, int] = {}
    rows = []
    for r in sorted(records, key=lambda r: r["video_id"]):
        video_id = str(r["video_id"])
        split = _assign_split(_parent_video_id(video_id), seed)
        counts[split] = counts.get(split, 0) + 1
        rows.append((video_id, split))
    if dry_run:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"[dry-run] split_info.csv: {len(rows)} dòng ({summary}) → {dest}")
        return counts
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["video_id", "split"])
        writer.writerows(rows)
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"split_info.csv: {len(rows)} dòng ({summary}) → {dest}")
    return counts


def _route_keypoints(records, project_root: Path, public_root: Path, *, copy, dry_run) -> dict:
    """Route .pose / *_preprocessed.npy đã có sẵn vào cam_front / cam_side."""
    counts = {"pose": 0, "npy": 0, "missing": 0}
    pose_search = [project_root / "Dataset" / "keypoints" / "raw", project_root / "Keypoint" / "raw"]
    npy_search = [project_root / "Dataset" / "keypoints" / "processed", project_root / "Keypoint" / "processed"]

    def _find(dirs, name):
        for base in dirs:
            for cand in (base / name, base / "cam_front" / name, base / "cam_side" / name):
                if cand.is_file():
                    return cand
        return None

    op = shutil.copy2 if copy else shutil.move
    for r in records:
        vid = str(r["video_id"])
        cam = "cam_side" if vid.endswith(_SIDE_SUFFIX) else "cam_front"
        for kind, name, search, key in (
            ("raw", f"{vid}.pose", pose_search, "pose"),
            ("processed", f"{vid}_preprocessed.npy", npy_search, "npy"),
        ):
            src = _find(search, name)
            dest_dir = public_root / "Keypoint" / kind / cam
            if src is None:
                if not (dest_dir / name).is_file():
                    counts["missing"] += 1
                continue
            if dry_run:
                print(f"[dry-run] {'copy' if copy else 'move'} {name} → {dest_dir}")
            else:
                dest_dir.mkdir(parents=True, exist_ok=True)
                op(str(src), str(dest_dir / name))
            counts[key] += 1
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--public-root", required=True, type=Path)
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=None,
        help="Thư mục chứa front_view.json/side_view.json (mặc định: "
        "<project-root>/Dataset/processed_videos/metadata).",
    )
    parser.add_argument(
        "--split-seed",
        default="vsl-qipedc",
        help="Seed chia train/val/test (đổi seed → chia khác, vẫn xác định).",
    )
    parser.add_argument("--keypoints", action="store_true", help="Route .pose/.npy đã có vào cam_*.")
    parser.add_argument("--copy", action="store_true", help="Khi --keypoints: copy thay vì move.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    public_root = args.public_root.resolve()
    metadata_dir = (
        args.metadata_dir or project_root / "Dataset" / "processed_videos" / "metadata"
    ).resolve()

    front = _load_view_json(metadata_dir / "front_view.json")
    side = _load_view_json(metadata_dir / "side_view.json")
    if not front and not side:
        print(
            f"Không tìm thấy front_view.json/side_view.json trong {metadata_dir}. "
            "Chạy `python -m qipedc2vsl400.convert` trước.",
            file=sys.stderr,
        )
        return 1

    print(f"Đọc metadata: front={len(front)} record, side={len(side)} record")

    # 1) cam_front.json / cam_side.json (transform từ converter output).
    pub_front = [_to_public_record(r) for r in front]
    pub_side = [_to_public_record(r) for r in side]
    meta_out = public_root / "Processed" / "metadata"
    _write_json(pub_front, meta_out / "cam_front.json", args.dry_run, "cam_front.json")
    if pub_side:
        _write_json(pub_side, meta_out / "cam_side.json", args.dry_run, "cam_side.json")

    # 2) labels.csv + gloss.csv (front + side gộp; gloss_id xác định).
    all_records = pub_front + pub_side
    gloss_index = _build_gloss_index(all_records)
    _write_labels_csv(all_records, gloss_index, public_root / "Processed" / "labels" / "labels.csv", args.dry_run)
    _write_gloss_csv(gloss_index, meta_out / "gloss.csv", args.dry_run)

    # 3) split_info.csv (chia xác định theo parent_video_id).
    _write_split_csv(all_records, meta_out / "split_info.csv", args.split_seed, args.dry_run)

    # 4) (tùy chọn) route keypoint đã có.
    if args.keypoints:
        counts = _route_keypoints(all_records, project_root, public_root, copy=args.copy, dry_run=args.dry_run)
        print(
            f"keypoint: .pose={counts['pose']} .npy={counts['npy']} thiếu={counts['missing']} "
            "(chưa trích ở repo training?)"
        )

    print(
        "\nXong. Lưu ý: cam_front.json/cam_side.json copy nguyên signer_id từ bước "
        "chia signer (qipedc2vsl400); labels.csv/gloss.csv/split_info.csv được sinh "
        "xác định từ tập gloss. Bản thân .pose/.npy do repo training tạo."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

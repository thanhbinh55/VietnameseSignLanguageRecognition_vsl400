from __future__ import annotations

import argparse
import logging
from pathlib import Path

from qipedc_video_preprocess.boundary_comparison import (
    compare_boundary_methods,
    format_case_table,
    format_summary_table,
    load_boundary_cases,
    recommend_method,
)
from qipedc_video_preprocess.config import PreprocessConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare OCR/overlay and pose boundary detection against Dataset/true_label."
    )
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument(
        "--true-label-dir",
        default="Dataset/true_label",
        help="Path to the ground-truth clip directory.",
    )
    parser.add_argument(
        "--methods",
        default="ocr,pose,ensemble,calibrated",
        help="Comma-separated methods to score.",
    )
    parser.add_argument(
        "--tolerance-seconds",
        type=float,
        default=None,
        help="Override the comparison tolerance (defaults to cfg.ensemble_tolerance_seconds).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Limit the number of per-case rows printed (0 disables the cap).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO logging while loading and scoring.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    cfg = PreprocessConfig(project_root=Path(args.project_root))
    true_label_dir = cfg.resolve(args.true_label_dir)
    methods = tuple(
        method.strip().lower()
        for method in args.methods.split(",")
        if method.strip()
    )
    cases = load_boundary_cases(true_label_dir, cfg)
    matches, summaries = compare_boundary_methods(
        cfg,
        cases,
        methods=methods,
        tolerance_seconds=args.tolerance_seconds,
    )

    limit = None if args.limit is not None and args.limit <= 0 else args.limit
    print(f"project_root={cfg.project_root}")
    print(f"true_label_dir={true_label_dir}")
    print(f"cases={len(cases)} methods={','.join(methods)}")
    print()
    print(format_summary_table(summaries, cfg.ensemble_tolerance_seconds if args.tolerance_seconds is None else args.tolerance_seconds))
    print()
    print(format_case_table(matches, limit=limit))
    recommendation = recommend_method(
        {method: summary for method, summary in summaries.items() if method != "calibrated"}
    )
    if recommendation is None:
        print()
        print("recommendation=unavailable")
    else:
        print()
        print(f"recommendation={recommendation}")
    if "calibrated" in summaries:
        print("calibrated_note=label_fitted_benchmark_only")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

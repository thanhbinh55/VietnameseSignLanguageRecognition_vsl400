from __future__ import annotations

from qipedc_video_preprocess.boundary_comparison import (
    BoundaryMatch,
    recommend_method,
    parse_true_label_name,
    summarize_method,
)


def test_parse_true_label_name_handles_front_and_side_variants():
    assert parse_true_label_name("W00144_c2_side.mp4") == ("W00144", 2, True)
    assert parse_true_label_name("D0093_side.mp4") == ("D0093", None, True)
    assert parse_true_label_name("W00738_c1.mp4") == ("W00738", 1, False)
    assert parse_true_label_name("W00738.mp4") == ("W00738", None, False)


def test_summary_and_recommendation_prefer_lower_error():
    ocr = summarize_method(
        "ocr",
        [
            BoundaryMatch(
                video_id="W00738",
                method="ocr",
                gt_boundaries=(72,),
                predicted_boundaries=(75,),
                matched_errors_frames=(3,),
                matched_errors_seconds=(0.1,),
                missed_boundaries=0,
                extra_boundaries=0,
                false_split=False,
                fps=30.0,
            )
        ],
        tolerance_seconds=0.3,
    )
    pose = summarize_method(
        "pose",
        [
            BoundaryMatch(
                video_id="W00738",
                method="pose",
                gt_boundaries=(72,),
                predicted_boundaries=(73,),
                matched_errors_frames=(1,),
                matched_errors_seconds=(0.0333333333,),
                missed_boundaries=0,
                extra_boundaries=0,
                false_split=False,
                fps=30.0,
            )
        ],
        tolerance_seconds=0.3,
    )

    recommendation = recommend_method({"ocr": ocr, "pose": pose})
    assert recommendation == "pose"
    assert ocr.cases == 1
    assert pose.within_tolerance == 1

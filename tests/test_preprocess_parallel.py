from __future__ import annotations

import logging
import importlib
import json
from pathlib import Path

from qipedc_video_preprocess import number_detector, preprocess
from qipedc_video_preprocess.config import PreprocessConfig
from qipedc_video_preprocess.segmenter import SegmentationResult, VariantSpan
from qipedc_video_preprocess.splitter import OutputClip
from qipedc_video_preprocess.video_probe import VideoProps


def test_config_accepts_four_parallel_workers_and_cuda_ocr():
    cfg = PreprocessConfig(
        project_root="D:/projects/metadata_VSL",
        parallel_workers=4,
        ocr_gpu=True,
        resume_existing=True,
    )

    assert cfg.parallel_workers == 4
    assert cfg.ocr_gpu is True
    assert cfg.resume_existing is True
    assert cfg.validate() == []


def test_config_rejects_non_positive_parallel_workers():
    cfg = PreprocessConfig(
        project_root="D:/projects/metadata_VSL",
        parallel_workers=0,
    )

    assert "parallel_workers must be in [1, 16]." in cfg.validate()


def test_run_output_jobs_uses_requested_process_count(monkeypatch):
    seen: dict[str, int] = {}

    class FakeExecutor:
        def __init__(self, *, max_workers):
            seen["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, fn, jobs, *, chunksize):
            seen["chunksize"] = chunksize
            return [fn(job) for job in jobs]

    monkeypatch.setattr(preprocess, "_process_output_clip", lambda job: job * 2)

    results = preprocess._run_output_jobs(
        [1, 2, 3],
        max_workers=4,
        executor_factory=FakeExecutor,
    )

    assert results == [2, 4, 6]
    assert seen == {"max_workers": 4, "chunksize": 1}


def test_default_detector_receives_cuda_setting(monkeypatch):
    captured = {}

    class FakeDetector:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    active_number_detector = importlib.import_module(
        "qipedc_video_preprocess.number_detector"
    )
    monkeypatch.setattr(active_number_detector, "EasyOcrNumberDetector", FakeDetector)
    cfg = PreprocessConfig(
        project_root="D:/projects/metadata_VSL",
        ocr_gpu=True,
    )

    detector = preprocess._make_default_detector(cfg, logging.getLogger("cuda-test"))

    assert isinstance(detector, FakeDetector)
    assert captured["gpu"] is True


def test_process_output_clip_returns_confident_front_and_side_labels(tmp_path, monkeypatch):
    cfg = PreprocessConfig(project_root=tmp_path)
    clip = OutputClip("W001", 1, "W001_c1.mp4", 0, 99)
    seg = SegmentationResult("W001", "multi", 2, (VariantSpan(1, 0, 99),), ())
    props = VideoProps(30.0, 100, 3.33, (1280, 720))
    job = preprocess.OutputWriteJob(Path("W001.mp4"), clip, seg, props, cfg)
    written_paths = []

    monkeypatch.setattr(preprocess, "_clip_span", lambda *args: VariantSpan(1, 0, 99))
    monkeypatch.setattr(
        preprocess,
        "_expand_views",
        lambda *args: [
            ("W001_c1.mp4", VariantSpan(1, 0, 39), False, "confident"),
            ("W001_c1_side.mp4", VariantSpan(1, 40, 99), True, "confident"),
        ],
    )
    monkeypatch.setattr(
        preprocess,
        "write_clip",
        lambda src, out, span, props, log: written_paths.append(out) or True,
    )

    result = preprocess._process_output_clip(job)

    assert result.written == 2
    assert result.manual_review is False
    assert [clip.out_filename for clip in result.label_clips] == [
        "W001_c1.mp4",
        "W001_c1_side.mp4",
    ]
    assert all("split_variants" in str(path) for path in written_paths)


def test_process_output_clip_routes_predicted_without_labels(tmp_path, monkeypatch):
    cfg = PreprocessConfig(project_root=tmp_path)
    clip = OutputClip("W001", 1, "W001_c1.mp4", 0, 99)
    seg = SegmentationResult(
        "W001", "multi", 2, (VariantSpan(1, 0, 99),), (), inferred=True
    )
    props = VideoProps(30.0, 100, 3.33, (1280, 720))
    job = preprocess.OutputWriteJob(Path("W001.mp4"), clip, seg, props, cfg)
    written_paths = []

    monkeypatch.setattr(preprocess, "_clip_span", lambda *args: VariantSpan(1, 0, 99))
    monkeypatch.setattr(
        preprocess,
        "_expand_views",
        lambda *args: [
            ("W001_c1.mp4", VariantSpan(1, 0, 99), False, "confident"),
        ],
    )
    monkeypatch.setattr(
        preprocess,
        "write_clip",
        lambda src, out, span, props, log: written_paths.append(out) or True,
    )

    result = preprocess._process_output_clip(job)

    assert result.written == 1
    assert result.label_clips == ()
    assert "split_variant_predicted" in str(written_paths[0])


def test_resume_existing_confident_clip_restores_label_without_work(tmp_path, monkeypatch):
    cfg = PreprocessConfig(
        project_root=tmp_path,
        resume_existing=True,
    )
    cfg.split_output_path.mkdir(parents=True)
    front = cfg.split_output_path / "W001_c1.mp4"
    side = cfg.split_output_path / "W001_c1_side.mp4"
    front.write_bytes(b"front")
    side.write_bytes(b"side")
    cfg.completion_path.mkdir(parents=True)
    (cfg.completion_path / "W001_c1.json").write_text(
        json.dumps(
            {
                "version": preprocess.OUTPUT_MANIFEST_VERSION,
                "video_id": "W001",
                "variant_index": 1,
                "outputs": [
                    {
                        "filename": "W001_c1.mp4",
                        "route": "confident",
                        "start_frame": 0,
                        "end_frame": 39,
                    },
                    {
                        "filename": "W001_c1_side.mp4",
                        "route": "confident",
                        "start_frame": 40,
                        "end_frame": 99,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(preprocess, "_output_is_readable", lambda path: True)
    clip = OutputClip("W001", 1, "W001_c1.mp4", 0, 99)
    seg = SegmentationResult("W001", "multi", 2, (VariantSpan(1, 0, 99),), ())
    props = VideoProps(30.0, 100, 3.33, (1280, 720))
    job = preprocess.OutputWriteJob(Path("W001.mp4"), clip, seg, props, cfg)

    result = preprocess._resume_output_result(job)

    assert result is not None
    assert result.written == 0
    assert [item.out_filename for item in result.label_clips] == [
        "W001_c1.mp4",
        "W001_c1_side.mp4",
    ]


def test_resume_existing_predicted_clip_has_no_label(tmp_path, monkeypatch):
    cfg = PreprocessConfig(project_root=tmp_path, resume_existing=True)
    cfg.inferred_review_path.mkdir(parents=True)
    predicted = cfg.inferred_review_path / "W001_c1.mp4"
    predicted.write_bytes(b"predicted")
    cfg.completion_path.mkdir(parents=True)
    (cfg.completion_path / "W001_c1.json").write_text(
        json.dumps(
            {
                "version": preprocess.OUTPUT_MANIFEST_VERSION,
                "video_id": "W001",
                "variant_index": 1,
                "outputs": [
                    {
                        "filename": "W001_c1.mp4",
                        "route": "predicted",
                        "start_frame": 0,
                        "end_frame": 99,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(preprocess, "_output_is_readable", lambda path: True)
    clip = OutputClip("W001", 1, "W001_c1.mp4", 0, 99)
    seg = SegmentationResult(
        "W001", "multi", 2, (VariantSpan(1, 0, 99),), (), inferred=True
    )
    props = VideoProps(30.0, 100, 3.33, (1280, 720))
    job = preprocess.OutputWriteJob(Path("W001.mp4"), clip, seg, props, cfg)

    result = preprocess._resume_output_result(job)

    assert result is not None
    assert result.label_clips == ()


def test_resume_does_not_trust_unmarked_front_file(tmp_path, monkeypatch):
    cfg = PreprocessConfig(project_root=tmp_path, resume_existing=True)
    cfg.split_output_path.mkdir(parents=True)
    (cfg.split_output_path / "W001_c1.mp4").write_bytes(b"front")
    monkeypatch.setattr(preprocess, "_output_is_readable", lambda path: True)
    clip = OutputClip("W001", 1, "W001_c1.mp4", 0, 99)
    seg = SegmentationResult("W001", "multi", 2, (VariantSpan(1, 0, 99),), ())
    props = VideoProps(30.0, 100, 3.33, (1280, 720))
    job = preprocess.OutputWriteJob(Path("W001.mp4"), clip, seg, props, cfg)

    assert preprocess._resume_output_result(job) is None


def test_resume_rejects_manifest_when_recorded_side_is_missing(tmp_path, monkeypatch):
    cfg = PreprocessConfig(project_root=tmp_path, resume_existing=True)
    cfg.split_output_path.mkdir(parents=True)
    (cfg.split_output_path / "W001_c1.mp4").write_bytes(b"front")
    cfg.completion_path.mkdir(parents=True)
    (cfg.completion_path / "W001_c1.json").write_text(
        json.dumps(
            {
                "version": preprocess.OUTPUT_MANIFEST_VERSION,
                "video_id": "W001",
                "variant_index": 1,
                "outputs": [
                    {
                        "filename": "W001_c1.mp4",
                        "route": "confident",
                        "start_frame": 0,
                        "end_frame": 39,
                    },
                    {
                        "filename": "W001_c1_side.mp4",
                        "route": "confident",
                        "start_frame": 40,
                        "end_frame": 99,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(preprocess, "_output_is_readable", lambda path: path.exists())
    clip = OutputClip("W001", 1, "W001_c1.mp4", 0, 99)
    seg = SegmentationResult("W001", "multi", 2, (VariantSpan(1, 0, 99),), ())
    props = VideoProps(30.0, 100, 3.33, (1280, 720))
    job = preprocess.OutputWriteJob(Path("W001.mp4"), clip, seg, props, cfg)

    assert preprocess._resume_output_result(job) is None
    assert not (cfg.completion_path / "W001_c1.json").exists()

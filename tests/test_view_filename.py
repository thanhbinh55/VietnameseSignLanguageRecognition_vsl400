"""Tests cho ``qipedc_video_preprocess.splitter.view_filename`` (THUẦN)."""

from __future__ import annotations

from qipedc_video_preprocess.splitter import view_filename


def test_front_keeps_name():
    assert view_filename("W00738", is_side=False) == "W00738.mp4"


def test_side_gets_suffix():
    assert view_filename("W00738", is_side=True) == "W00738_side.mp4"


def test_side_on_variant_name():
    assert view_filename("W00738_c2", is_side=True) == "W00738_c2_side.mp4"
    assert view_filename("W00738_c2", is_side=False) == "W00738_c2.mp4"

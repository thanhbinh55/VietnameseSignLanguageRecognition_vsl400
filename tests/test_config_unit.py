"""Unit tests for ``qipedc_video_preprocess.config`` â€” defaults & isolation.

Phá»§ hai khÃ­a cáº¡nh cá»§a Requirement 1:

* **Req 1.6** â€” má»i Ä‘Æ°á»ng dáº«n Ä‘áº§u ra máº·c Ä‘á»‹nh resolve vá» á»• ``D:`` (khÃ´ng bao giá»
  ``C:``) khi ``project_root`` náº±m trÃªn á»• ``D:``.
* **Req 1.1** â€” cÃ´ng Ä‘oáº¡n tÃ¡ch video Ä‘Æ°á»£c cÃ´ láº­p trong package
  ``qipedc_video_preprocess`` vÃ  **khÃ´ng** import/tham chiáº¿u hay Ä‘áº·t module bÃªn
  trong package ``qipedc2vsl400``.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

from qipedc_video_preprocess.config import PreprocessConfig

# project_root giáº£ láº­p náº±m trÃªn á»• D: (test thuáº§n logic Ä‘Æ°á»ng dáº«n, khÃ´ng cháº¡m Ä‘Ä©a).
PROJECT_ROOT = Path("D:/projects/metadata_VSL")

# ThÆ° má»¥c package nguá»“n cá»§a cÃ´ng Ä‘oáº¡n vÃ  cá»§a pipeline metadata hiá»‡n cÃ³.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
_PREPROCESS_PKG_DIR = _SRC_DIR / "qipedc_video_preprocess"
_LEGACY_PKG_DIR = _SRC_DIR / "qipedc2vsl400"

# TÃªn cÃ¡c module thuá»™c cÃ´ng Ä‘oáº¡n tÃ¡ch video (theo design.md / tasks.md).
_PREPROCESS_MODULE_NAMES = {
    "boundary_calibration",
    "boundary_comparison",
    "config",
    "discovery",
    "number_detector",
    "pose_boundary",
    "video_probe",
    "segmenter",
    "splitter",
    "label_writer",
    "run_logger",
    "preprocess",
}


def make_cfg() -> PreprocessConfig:
    return PreprocessConfig(project_root=PROJECT_ROOT)


# --------------------------------------------------------------------------- #
# Req 1.6 â€” má»i Ä‘Æ°á»ng dáº«n Ä‘áº§u ra máº·c Ä‘á»‹nh resolve trÃªn á»• D:
# --------------------------------------------------------------------------- #
def test_default_output_paths_resolve_on_d_drive():
    """Má»i Ä‘Æ°á»ng dáº«n Ä‘áº§u ra máº·c Ä‘á»‹nh pháº£i resolve lÃªn á»• D:, khÃ´ng bao giá» C:."""
    cfg = make_cfg()
    output_paths = {
        "split_output_dir": cfg.resolve(cfg.split_output_dir),
        "new_labels_path": cfg.resolve(cfg.new_labels_path),
        "log_dir": cfg.resolve(cfg.log_dir),
        "ocr_models_dir": cfg.resolve(cfg.ocr_models_dir),
    }
    for name, path in output_paths.items():
        assert path.is_absolute(), f"{name} pháº£i lÃ  Ä‘Æ°á»ng dáº«n tuyá»‡t Ä‘á»‘i"
        assert path.drive.upper() == "D:", f"{name} pháº£i náº±m trÃªn á»• D: (Ä‘Æ°á»£c {path.drive!r})"
        assert path.drive.upper() != "C:", f"{name} khÃ´ng Ä‘Æ°á»£c náº±m trÃªn á»• C:"


def test_default_output_paths_under_project_root():
    """Má»i Ä‘Æ°á»ng dáº«n Ä‘áº§u ra máº·c Ä‘á»‹nh náº±m bÃªn trong cÃ¢y project_root."""
    cfg = make_cfg()
    root = PROJECT_ROOT.resolve()
    for raw in (cfg.split_output_dir, cfg.new_labels_path, cfg.log_dir, cfg.ocr_models_dir):
        resolved = cfg.resolve(raw)
        # KhÃ´ng nÃ©m ValueError nghÄ©a lÃ  náº±m trong cÃ¢y project_root.
        resolved.relative_to(root)


def test_convenience_path_properties_on_d_drive():
    """CÃ¡c property tiá»‡n Ã­ch cÅ©ng resolve lÃªn á»• D: táº¡i Ä‘Ãºng vá»‹ trÃ­ mong Ä‘á»£i."""
    cfg = make_cfg()
    root = PROJECT_ROOT.resolve()
    assert cfg.split_output_path == (root / cfg.split_output_dir).resolve()
    assert cfg.new_labels_full_path == (root / cfg.new_labels_path).resolve()
    assert cfg.log_path == (root / cfg.log_dir).resolve()
    assert cfg.ocr_models_path == (root / cfg.ocr_models_dir).resolve()
    for path in (cfg.split_output_path, cfg.new_labels_full_path, cfg.log_path, cfg.ocr_models_path):
        assert path.drive.upper() == "D:"


def test_default_config_validates_clean_on_d_drive():
    """Cáº¥u hÃ¬nh máº·c Ä‘á»‹nh trÃªn á»• D: khÃ´ng cÃ³ lá»—i (táº¥t cáº£ output trÃªn D:, trong cÃ¢y)."""
    cfg = make_cfg()
    assert cfg.validate() == []


def test_default_boundary_tolerances_are_split_by_task():
    cfg = make_cfg()
    assert cfg.ensemble_tolerance_seconds == pytest.approx(0.6)
    assert cfg.view_ensemble_tolerance_seconds == pytest.approx(0.3)
    assert cfg.view_predicted_tolerance_seconds == pytest.approx(0.6)
    assert cfg.multiview_diff_threshold == pytest.approx(0.01)


def test_video_search_paths_resolve_on_d_drive():
    cfg = make_cfg()
    paths = cfg.video_search_paths()
    assert len(paths) == len(cfg.video_search_dirs)
    assert all(p.drive.upper() == "D:" for p in paths)


# --------------------------------------------------------------------------- #
# Req 1.1 â€” cÃ´ láº­p: khÃ´ng import/tham chiáº¿u/Ä‘áº·t module trong qipedc2vsl400
# --------------------------------------------------------------------------- #
def _imported_module_names(source: str) -> set[str]:
    """TrÃ­ch tÃªn module gá»‘c tá»« má»i cÃ¢u lá»‡nh import trong mÃ£ nguá»“n."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                names.add(node.module.split(".")[0])
    return names


def test_config_module_does_not_import_legacy_package():
    """config.py khÃ´ng import qipedc2vsl400 (phÃ¢n tÃ­ch AST)."""
    source = (_PREPROCESS_PKG_DIR / "config.py").read_text(encoding="utf-8")
    assert "qipedc2vsl400" not in _imported_module_names(source)


def test_config_module_source_has_no_legacy_reference():
    """config.py khÃ´ng nháº¯c tá»›i qipedc2vsl400 trong báº¥t ká»³ cÃ¢u lá»‡nh import nÃ o."""
    source = (_PREPROCESS_PKG_DIR / "config.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            segment = ast.get_source_segment(source, node) or ""
            assert "qipedc2vsl400" not in segment


@pytest.mark.parametrize("module_name", sorted(_PREPROCESS_MODULE_NAMES))
def test_no_preprocess_module_imports_legacy_package(module_name):
    """KhÃ´ng module nÃ o cá»§a cÃ´ng Ä‘oáº¡n import qipedc2vsl400 (cÃ´ láº­p, Req 1.1)."""
    module_file = _PREPROCESS_PKG_DIR / f"{module_name}.py"
    if not module_file.exists():
        pytest.skip(f"{module_name}.py chÆ°a Ä‘Æ°á»£c hiá»‡n thá»±c")
    source = module_file.read_text(encoding="utf-8")
    assert "qipedc2vsl400" not in _imported_module_names(source), (
        f"{module_name}.py khÃ´ng Ä‘Æ°á»£c import qipedc2vsl400"
    )


def test_importing_config_does_not_load_legacy_package():
    """Import config (má»›i) khÃ´ng kÃ©o theo qipedc2vsl400 vÃ o sys.modules."""
    # Loáº¡i bá» má»i tÃ n dÆ° Ä‘á»ƒ phÃ©p kiá»ƒm tra pháº£n Ã¡nh Ä‘Ãºng quan há»‡ phá»¥ thuá»™c.
    for name in list(sys.modules):
        if name == "qipedc2vsl400" or name.startswith("qipedc2vsl400."):
            del sys.modules[name]
    for name in list(sys.modules):
        if name == "qipedc_video_preprocess" or name.startswith("qipedc_video_preprocess."):
            del sys.modules[name]

    importlib.import_module("qipedc_video_preprocess.config")

    leaked = [n for n in sys.modules if n == "qipedc2vsl400" or n.startswith("qipedc2vsl400.")]
    assert leaked == [], f"Import config khÃ´ng Ä‘Æ°á»£c náº¡p qipedc2vsl400: {leaked}"


def test_preprocess_modules_live_in_dedicated_package():
    """Má»i module cÃ´ng Ä‘oáº¡n Ä‘Æ°á»£c khai bÃ¡o trong package qipedc_video_preprocess (Req 1.1)."""
    assert _PREPROCESS_PKG_DIR.is_dir(), "package qipedc_video_preprocess pháº£i tá»“n táº¡i"
    for module_name in _PREPROCESS_MODULE_NAMES:
        assert (_PREPROCESS_PKG_DIR / f"{module_name}.py").exists(), (
            f"{module_name}.py pháº£i náº±m trong qipedc_video_preprocess/"
        )


def test_preprocess_specific_modules_absent_from_legacy_package():
    """CÃ¡c module Ä‘áº·c thÃ¹ cÃ´ng Ä‘oáº¡n (vd splitter, segmenter) khÃ´ng Ä‘Æ°á»£c Ä‘áº·t trong qipedc2vsl400/."""
    if not _LEGACY_PKG_DIR.is_dir():
        pytest.skip("qipedc2vsl400 khÃ´ng tá»“n táº¡i")
    # 'config' vÃ  'video_probe' lÃ  tÃªn chung tá»“n táº¡i há»£p lá»‡ vÃ  Ä‘á»™c láº­p á»Ÿ cáº£ hai
    # package (má»—i package cÃ³ báº£n hiá»‡n thá»±c riÃªng); viá»‡c cÃ´ láº­p thá»±c sá»± (khÃ´ng
    # import chÃ©o) Ä‘Ã£ Ä‘Æ°á»£c cÃ¡c test quÃ©t AST á»Ÿ trÃªn báº£o Ä‘áº£m. á»ž Ä‘Ã¢y chá»‰ kiá»ƒm cÃ¡c
    # module mang tÃ­nh Ä‘áº·c thÃ¹ cá»§a cÃ´ng Ä‘oáº¡n tÃ¡ch video.
    preprocess_specific = _PREPROCESS_MODULE_NAMES - {"config", "video_probe"}
    for module_name in preprocess_specific:
        assert not (_LEGACY_PKG_DIR / f"{module_name}.py").exists(), (
            f"{module_name}.py khÃ´ng Ä‘Æ°á»£c khai bÃ¡o bÃªn trong qipedc2vsl400/"
        )


def test_legacy_package_does_not_reference_preprocess_package():
    """Package qipedc2vsl400 khÃ´ng tham chiáº¿u cÃ´ng Ä‘oáº¡n má»›i (cÃ´ láº­p hai chiá»u)."""
    if not _LEGACY_PKG_DIR.is_dir():
        pytest.skip("qipedc2vsl400 khÃ´ng tá»“n táº¡i")
    for py_file in _LEGACY_PKG_DIR.glob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        assert "qipedc_video_preprocess" not in _imported_module_names(source), (
            f"{py_file.name} (legacy) khÃ´ng Ä‘Æ°á»£c import qipedc_video_preprocess"
        )

from __future__ import annotations

from conftest import REPO_ROOT


REPORT_DIR = REPO_ROOT / "report"
REPORT_SUMMARY = REPORT_DIR / "REPORT_LATEX_UPDATE_SUMMARY.md"
TENNIS_SUMMARY = REPORT_DIR / "TENNIS_EXTENSION_LATEX_UPDATE_SUMMARY.md"


def test_report_summary_uses_m7_audit_as_final_conflict_detection() -> None:
    text = REPORT_SUMMARY.read_text(encoding="utf-8")

    assert "M7 LLM-agent reflection audit" in text
    assert "tiser x tiser M7 conflict-naming | 0.042" in text
    assert "base x tiser M7 conflict-naming | 0.074" in text
    assert "M6 diagnostic, not final H2 metric" in text
    assert "Conflict-detection is described as a lexical proxy throughout" not in text
    assert "reflection mention rate 3.8%" not in text


def test_tennis_report_summary_lists_current_result_database() -> None:
    text = TENNIS_SUMMARY.read_text(encoding="utf-8")

    assert "base_qwen_standard_test224" in text
    assert "base_qwen_tiser_test224" in text
    assert "tennis_only_full600_test224" in text
    assert "original_tiser_qwen7b_test224" in text
    assert "tennis_from_tiser_e2_lr0.0002_bs4_ga4_r16_a32_d0p05_20260616_104036_011" in text
    assert "0.732" in text
    assert "0.856" in text
    assert "final E0-E3 adapter-comparison metrics are missing" not in text
    assert "TODO metric placeholders" not in text
    assert "Original TISER-adapter transfer and mixed tennis+TISER replay were not run" not in text

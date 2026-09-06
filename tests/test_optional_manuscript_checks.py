from pathlib import Path

import pytest

from scripts.audit_rotating_colloids_capillary_prl import manuscript_status


def test_default_does_not_read_manuscript(monkeypatch):
    def unexpected_read(*args, **kwargs):
        pytest.fail("Default numerical audit must not read manuscript sources")

    monkeypatch.setattr(Path, "read_text", unexpected_read)
    status = manuscript_status(None)
    assert status["manuscript_check_status"] == "not_requested"
    assert status["all_language_gates_passed"] is None
    assert status["supplemental_figure_references"] is None


def test_explicit_missing_sources_fail(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing external manuscript sources"):
        manuscript_status(tmp_path)


@pytest.mark.parametrize("wording,passed", [("Finite-window memory.", True), ("Permanent memory.", False)])
def test_external_sources_are_checked(tmp_path, wording, passed):
    (tmp_path / "rotating_colloids_prl_capillary.tex").write_text(
        wording + r" See Fig.~S1 and Table~SI.", encoding="utf-8"
    )
    (tmp_path / "rotating_colloids_prl_capillary_supplement.tex").write_text(
        r"\label{fig:supp_example}\label{tab:supp_example}", encoding="utf-8"
    )
    status = manuscript_status(tmp_path)
    assert status["manuscript_check_status"] == "checked"
    assert status["all_language_gates_passed"] is passed
    assert status["supplemental_figure_references"]["all_references_resolve"]

"""Testes de comportamento, determinismo e escrita atômica da exportação."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from research_atlas.evaluation import export as export_module
from research_atlas.evaluation.export import (
    DEFAULT_OUTPUT_DIR,
    EXPORTED_CSV_FILES,
    export_evaluation_artifacts,
    main,
)
from research_atlas.evaluation.pipeline import PROCESSED_DIR, sha256_file


def _dir_bytes(directory: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def test_importing_export_module_has_no_filesystem_side_effects():
    code = (
        "import pathlib, pandas as pd; "
        "pd.DataFrame.to_csv = lambda *a, **k: (_ for _ in ()).throw(AssertionError('to_csv chamado no import')); "
        "pathlib.Path.write_text = lambda *a, **k: (_ for _ in ()).throw(AssertionError('write_text chamado no import')); "
        "import research_atlas.evaluation.export"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROCESSED_DIR.parents[1],
        env={
            **os.environ,
            "PYTHONPATH": str(PROCESSED_DIR.parents[1] / "src"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "bad_target",
    [
        PROCESSED_DIR,
        PROCESSED_DIR.parent,
        PROCESSED_DIR / "embeddings",
        PROCESSED_DIR / "embeddings/bge-small-en-v1.5-v1",
        PROCESSED_DIR / "clustering",
        PROCESSED_DIR / "clustering/v1",
        PROCESSED_DIR / "tfidf_baseline",
        PROCESSED_DIR / "tfidf_baseline/v1",
        PROCESSED_DIR / "audit",
    ],
)
def test_protected_output_directories_are_rejected(bad_target):
    with pytest.raises(ValueError, match="Diretório de saída"):
        export_evaluation_artifacts(output_dir=bad_target)


def test_two_consecutive_exports_are_byte_identical(tmp_path):
    first_dir = tmp_path / "run_1"
    second_dir = tmp_path / "run_2"

    main(["--output-dir", str(first_dir)])
    second = export_evaluation_artifacts(output_dir=second_dir)
    export_evaluation_artifacts(output_dir=first_dir)

    first_bytes = _dir_bytes(first_dir)
    second_bytes = _dir_bytes(second_dir)

    assert set(first_bytes) == {*EXPORTED_CSV_FILES, "manifest.json"}
    assert first_bytes == second_bytes

    if DEFAULT_OUTPUT_DIR.is_dir():
        assert first_bytes == _dir_bytes(DEFAULT_OUTPUT_DIR)

    for filename, expected_hash in second["manifest"]["files_sha256"].items():
        assert sha256_file(second_dir / filename) == expected_hash


def test_failed_export_leaves_no_partial_artifacts(tmp_path, monkeypatch):
    target_dir = tmp_path / "evaluation/v1"
    calls = 0
    original_write_csv = export_module._write_csv

    def flaky_write_csv(frame, path, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("Falha simulada de escrita")
        return original_write_csv(frame, path, **kwargs)

    monkeypatch.setattr(
        export_module,
        "build_evaluation_table",
        lambda *args, **kwargs: export_module.pd.DataFrame(
            {"n_articles": [10], "representation": ["tfidf"], "scope": ["full"]}
        ),
    )
    monkeypatch.setattr(
        export_module,
        "load_concordance_analysis",
        lambda *args, **kwargs: {
            "seed_stability": export_module.pd.DataFrame(),
            "concordance": export_module.pd.DataFrame(
                {"valid": export_module.pd.Series(dtype="boolean")}
            ),
            "full_ari_matrix": export_module.pd.DataFrame(),
            "identical_pairs": export_module.pd.DataFrame(),
            "reconciliation": export_module.pd.DataFrame(),
        },
    )
    monkeypatch.setattr(export_module, "_write_csv", flaky_write_csv)

    with pytest.raises(OSError, match="Falha simulada"):
        export_evaluation_artifacts(
            output_dir=target_dir,
            verify_reference=False,
        )

    assert not target_dir.exists()
    leftovers = list(target_dir.parent.iterdir())
    assert leftovers == []


def test_failed_overwrite_preserves_existing_artifacts(tmp_path, monkeypatch):
    target_dir = tmp_path / "evaluation/v1"
    target_dir.mkdir(parents=True)
    sentinel = target_dir / "evaluation_metrics.csv"
    sentinel.write_text("config_id\nkmeans_k8\n", encoding="utf-8")
    before = _dir_bytes(target_dir)

    monkeypatch.setattr(
        export_module,
        "build_evaluation_table",
        lambda *args, **kwargs: export_module.pd.DataFrame(
            {"n_articles": [10], "representation": ["tfidf"], "scope": ["full"]}
        ),
    )
    monkeypatch.setattr(
        export_module,
        "load_concordance_analysis",
        lambda *args, **kwargs: {
            "seed_stability": export_module.pd.DataFrame(),
            "concordance": export_module.pd.DataFrame(
                {"valid": export_module.pd.Series(dtype="boolean")}
            ),
            "full_ari_matrix": export_module.pd.DataFrame(),
            "identical_pairs": export_module.pd.DataFrame(),
            "reconciliation": export_module.pd.DataFrame(),
        },
    )

    def fail_manifest(*args, **kwargs):
        raise RuntimeError("Falha ao construir manifesto")

    monkeypatch.setattr(export_module, "_build_manifest", fail_manifest)

    with pytest.raises(RuntimeError, match="Falha ao construir manifesto"):
        export_evaluation_artifacts(
            output_dir=target_dir,
            verify_reference=False,
        )

    assert _dir_bytes(target_dir) == before
    sibling_names = [p.name for p in target_dir.parent.iterdir()]
    assert sibling_names == ["v1"]


def test_reconciliation_mismatch_aborts_before_writing(tmp_path, monkeypatch):
    target_dir = tmp_path / "evaluation/v1"

    monkeypatch.setattr(
        export_module,
        "build_evaluation_table",
        lambda *args, **kwargs: export_module.pd.DataFrame(
            {"n_articles": [10], "representation": ["tfidf"], "scope": ["full"]}
        ),
    )
    monkeypatch.setattr(
        export_module,
        "reconcile_with_reference",
        lambda *args, **kwargs: export_module.pd.DataFrame({"match": [False]}),
    )

    with pytest.raises(ValueError, match="Divergência na reconciliação"):
        export_evaluation_artifacts(output_dir=target_dir)

    assert not target_dir.exists()

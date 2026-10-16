"""Tests d'intégration : agent complet (graphe LangGraph) sur données réelles et synthétiques."""

import json
from pathlib import Path

import pandas as pd
import pytest

from agents.etl import ETLOptions, ETLRequest, run, run_with_data
from agents.etl.agent import _numbers_grounded
from agents.etl.models import ETLReport
from agents.profiler.models import ProfilingReport, SourceConfig, TableProfile

ROOT = Path(__file__).resolve().parents[2]
TELCO_CSV = ROOT / "data" / "telco.csv"
PROFILE_JSON = Path(__file__).parent / "fixtures" / "profiler_output.json"

needs_telco = pytest.mark.skipif(not TELCO_CSV.exists(), reason="data/telco.csv absent")


def _profile_for_tables(*names: str) -> ProfilingReport:
    return ProfilingReport(
        business_goal="Analyser le churn",
        tables=[TableProfile(name=n, source_name=n, row_count=0, column_count=0) for n in names],
    )


def _request(csv_path, tmp_path, name="t", **options) -> ETLRequest:
    options.setdefault("output_dir", str(tmp_path / "out"))
    options.setdefault("use_llm", False)
    return ETLRequest(
        profiling_report=_profile_for_tables(name),
        sources=[SourceConfig(name=name, type="csv", uri=str(csv_path))],
        options=ETLOptions(**options),
    )


# ------------------------------------------------------------------ données réelles
@pytest.fixture(scope="module")
def telco_result(tmp_path_factory):
    out = tmp_path_factory.mktemp("telco_out")
    profile = ProfilingReport.model_validate_json(PROFILE_JSON.read_text(encoding="utf-8"))
    request = ETLRequest(
        profiling_report=profile,
        sources=[SourceConfig(name="telco", type="csv", uri=str(TELCO_CSV))],
        options=ETLOptions(preset="telco", output_dir=str(out), use_llm=False),
    )
    report, frames = run_with_data(request)
    return report, frames, out


@needs_telco
def test_telco_end_to_end_status_and_shape(telco_result):
    report, frames, _ = telco_result
    assert report.status == "success", report.errors
    table = report.tables[0]
    assert (table.rows_in, table.rows_out) == (7043, 7043)  # aucune ligne perdue
    assert table.columns_out == 33
    assert frames["telco"].shape == (7043, 33)


@needs_telco
def test_telco_total_charges_fixed(telco_result):
    report, frames, _ = telco_result
    df = frames["telco"]
    assert pd.api.types.is_float_dtype(df["total_charges"])
    assert df["total_charges"].isna().sum() == 0
    zero_tenure = df[df["tenure_months"] == 0]
    assert len(zero_tenure) == 11 and (zero_tenure["total_charges"] == 0).all()
    rules = {a.rule: a.rows_affected for a in report.actions if a.column == "Total Charges"}
    assert rules["blank_to_null"] == 11 and rules["conditional_fill"] == 11


@needs_telco
def test_telco_churn_reason_is_structural_not_imputed(telco_result):
    report, frames, _ = telco_result
    df = frames["telco"]
    assert (df["churn_reason"] == "Not applicable").sum() == 5174
    assert ((df["churn_reason"] == "Not applicable") == (df["churn_label"] == "No")).all()
    resolution = next(r for r in report.issue_resolutions
                      if r.issue_id == "telco.Churn Reason.missing")
    assert resolution.status == "resolved" and resolution.action_ids


@needs_telco
def test_telco_quality_improves_and_checks_pass(telco_result):
    report, _, _ = telco_result
    table = report.tables[0]
    assert table.quality_after.overall > table.quality_before.overall
    assert table.quality_after.overall == 100.0
    assert table.quality_before.text_numeric_columns == ["Total Charges"]
    blocking = [c for c in report.validation_checks if not c.passed and c.severity != "low"]
    assert blocking == []


@needs_telco
def test_telco_pii_handling_and_role_corrections(telco_result):
    report, frames, _ = telco_result
    cols = {c.original_name: c for c in report.tables[0].columns}
    assert cols["CustomerID"].pii_treatment == "kept" and cols["CustomerID"].semantic_role == "id"
    for name in ("Lat Long", "Latitude", "Longitude", "Phone Service"):
        assert cols[name].pii_treatment == "false_positive_ignored"
    assert cols["Monthly Charges"].semantic_role == "measure"
    assert cols["Monthly Charges"].role_corrected_from == "date"
    assert cols["Zip Code"].semantic_role == "id"
    assert frames["telco"]["customer_id"].is_unique


@needs_telco
def test_telco_outputs_written_and_report_roundtrips(telco_result):
    report, _, out = telco_result
    csv_path = out / "telco_clean.csv"
    assert csv_path.exists() and report.tables[0].output_path == str(csv_path)
    reloaded = pd.read_csv(csv_path)
    assert reloaded.shape == (7043, 33) and reloaded["total_charges"].dtype == "float64"
    saved = ETLReport.model_validate_json((out / "etl_report.json").read_text(encoding="utf-8"))
    assert saved.status == report.status and len(saved.actions) == len(report.actions)


@needs_telco
def test_telco_every_action_is_explainable(telco_result):
    report, _, _ = telco_result
    assert report.actions
    for action in report.actions:
        assert action.rationale and action.before and action.after


@needs_telco
def test_etl_is_idempotent(tmp_path):
    """Nettoyer deux fois ne doit rien changer la deuxième fois."""
    first = _request(TELCO_CSV, tmp_path, name="telco", preset="telco",
                     standardize_column_names=False, output_dir=str(tmp_path / "o1"))
    first.profiling_report = ProfilingReport.model_validate_json(
        PROFILE_JSON.read_text(encoding="utf-8"))
    _, frames1 = run_with_data(first)

    cleaned = tmp_path / "clean.csv"
    frames1["telco"].to_csv(cleaned, index=False)
    second = _request(cleaned, tmp_path, name="telco", preset="telco",
                      standardize_column_names=False, output_dir=str(tmp_path / "o2"))
    report2, frames2 = run_with_data(second)

    changing = [a for a in report2.actions if a.rows_affected > 0]
    assert changing == []
    pd.testing.assert_frame_equal(frames1["telco"], frames2["telco"], check_dtype=False)


@needs_telco
def test_input_file_is_never_modified(tmp_path):
    before = TELCO_CSV.read_bytes()
    run(_request(TELCO_CSV, tmp_path, name="telco", preset="telco"))
    assert TELCO_CSV.read_bytes() == before


# ------------------------------------------------------------------ données synthétiques
def _write_dirty_csv(path: Path) -> None:
    rows = []
    for i in range(30):
        rows.append({"id": f"A{i:03d}", "amount": str(10 + i), "city": "Tunis" if i % 2 else "tunis",
                     "email": f"user{i}@mail.com"})
    rows[3]["amount"] = " "
    rows[5]["city"] = None
    df = pd.DataFrame(rows)
    df = pd.concat([df, df.iloc[[7]]], ignore_index=True)  # doublon exact
    df.to_csv(path, index=False)


def test_synthetic_dirty_table_is_cleaned_and_traced(tmp_path):
    csv = tmp_path / "dirty.csv"
    _write_dirty_csv(csv)
    request = _request(csv, tmp_path, business_keys={"t": ["id"]})
    report, frames = run_with_data(request)
    df = frames["t"]

    assert report.tables[0].rows_in == 31 and report.tables[0].rows_out == 30
    assert pd.api.types.is_numeric_dtype(df["amount"]) and df["amount"].isna().sum() == 0
    # 15 "tunis" vs 14 "Tunis" : les variantes de casse fusionnent vers la plus fréquente,
    # et la ville manquante reçoit la modalité explicite "Unknown".
    assert set(df["city"]) == {"tunis", "Unknown"}
    assert df["id"].is_unique
    assert not any("@" in str(v) for v in df["email"])  # PII pseudonymisée
    assert next(c for c in report.validation_checks if c.check == "row_reconciliation").passed
    # Défaut du salage par défaut : signalé, pas caché.
    assert any(i.rule == "pii_default_salt" for i in report.remaining_issues)
    assert report.status == "warning"


def test_pii_salt_from_environment_removes_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("BIFLOW_PII_SALT", "secret-salt")
    csv = tmp_path / "dirty.csv"
    _write_dirty_csv(csv)
    report = run(_request(csv, tmp_path, business_keys={"t": ["id"]}))
    assert not any(i.rule == "pii_default_salt" for i in report.remaining_issues)


def test_key_duplicates_are_flagged_not_deleted_by_default(tmp_path):
    csv = tmp_path / "k.csv"
    pd.DataFrame({"id": [1, 1, 2, 3, 4, 5], "v": ["a", "b", "c", "d", "e", "f"]}).to_csv(
        csv, index=False)
    report, frames = run_with_data(_request(csv, tmp_path, business_keys={"t": ["id"]}))
    assert len(frames["t"]) == 6
    assert any(a.rule == "flag_key_duplicates" for a in report.actions)
    assert any(i.rule == "business_key_unique" and i.severity == "high"
               for i in report.remaining_issues)
    assert report.status == "warning"


def test_key_duplicates_removed_when_strategy_set(tmp_path):
    csv = tmp_path / "k.csv"
    pd.DataFrame({"id": [1, 1, 2, 3, 4, 5], "v": list("abcdef")}).to_csv(csv, index=False)
    report, frames = run_with_data(_request(
        csv, tmp_path, business_keys={"t": ["id"]}, key_duplicate_strategy="keep_first"))
    assert frames["t"]["v"].tolist()[:2] == ["a", "c"] and len(frames["t"]) == 5


def test_write_outputs_false_creates_no_files(tmp_path):
    csv = tmp_path / "k.csv"
    pd.DataFrame({"id": range(10), "v": range(10)}).to_csv(csv, index=False)
    report = run(_request(csv, tmp_path, write_outputs=False))
    assert not (tmp_path / "out").exists() and report.tables[0].output_path is None


# ------------------------------------------------------------------ robustesse
def test_missing_source_file_returns_error_report_without_raising(tmp_path):
    report = run(_request(tmp_path / "does_not_exist.csv", tmp_path))
    assert report.status == "error" and report.tables == []
    assert any("does_not_exist" in e or "chargement" in e.lower() for e in report.errors)
    assert "n'a pas abouti" in report.llm_summary


def test_one_bad_source_does_not_block_the_others(tmp_path):
    good = tmp_path / "good.csv"
    pd.DataFrame({"id": range(10), "v": range(10)}).to_csv(good, index=False)
    request = ETLRequest(
        profiling_report=_profile_for_tables("good", "bad"),
        sources=[SourceConfig(name="good", type="csv", uri=str(good)),
                 SourceConfig(name="bad", type="csv", uri=str(tmp_path / "nope.csv"))],
        options=ETLOptions(output_dir=str(tmp_path / "o"), use_llm=False),
    )
    report = run(request)
    assert [t.name for t in report.tables] == ["good"]
    assert report.status == "warning" and report.errors


def test_source_without_profile_still_runs_with_heuristics(tmp_path):
    csv = tmp_path / "k.csv"
    pd.DataFrame({"id": range(10), "v": range(10)}).to_csv(csv, index=False)
    request = ETLRequest(
        profiling_report=_profile_for_tables("other"),
        sources=[SourceConfig(name="t", type="csv", uri=str(csv))],
        options=ETLOptions(output_dir=str(tmp_path / "o"), use_llm=False),
    )
    report = run(request)
    assert report.tables and any("absente du rapport" in e for e in report.errors)
    assert report.status == "warning"


def test_empty_sources_are_rejected_by_the_schema():
    with pytest.raises(Exception):
        ETLRequest(profiling_report=_profile_for_tables("t"), sources=[])


# ------------------------------------------------------------------ garde-fou LLM
def test_llm_numbers_guard():
    source = json.dumps({"rows_in": 7043, "overall": 95.832, "x": 5174})
    assert _numbers_grounded("7043 lignes, score 95.832, 5174 corrigées", source)
    assert not _numbers_grounded("8000 lignes ont été corrigées", source)
    assert _numbers_grounded("Une seule table et 3 colonnes", source)  # chiffres isolés tolérés


# ------------------------------------------------------------------ chemin LLM (faux LLM)
class _FakeLLM:
    def __init__(self, text: str | Exception):
        self.text = text

    def invoke(self, _messages):
        if isinstance(self.text, Exception):
            raise self.text
        return type("R", (), {"content": self.text})()


def _small_request(tmp_path, use_llm=True):
    csv = tmp_path / "k.csv"
    pd.DataFrame({"id": range(10), "v": range(10)}).to_csv(csv, index=False)
    return _request(csv, tmp_path, use_llm=use_llm)


def test_llm_summary_is_used_when_numbers_are_grounded(tmp_path, monkeypatch):
    import agents.etl.agent as agent
    monkeypatch.setattr(agent, "_load_llm", lambda: _FakeLLM("Table t : 10 lignes en entrée, 10 en sortie."))
    report = run(_small_request(tmp_path))
    assert report.llm_summary.startswith("Table t : 10 lignes en entrée")


def test_llm_summary_with_invented_number_falls_back(tmp_path, monkeypatch):
    import agents.etl.agent as agent
    monkeypatch.setattr(agent, "_load_llm", lambda: _FakeLLM("Nous avons corrigé 99999 lignes."))
    report = run(_small_request(tmp_path))
    assert "99999" not in report.llm_summary and "score qualité" in report.llm_summary


def test_llm_failure_falls_back_to_deterministic_summary(tmp_path, monkeypatch):
    import agents.etl.agent as agent
    monkeypatch.setattr(agent, "_load_llm", lambda: _FakeLLM(RuntimeError("ollama down")))
    report = run(_small_request(tmp_path))
    assert report.status == "success" and "score qualité" in report.llm_summary


# ------------------------------------------------------------------ 2e dataset (usage mobile)
def _usage_frame() -> pd.DataFrame:
    n = 40
    df = pd.DataFrame({
        "subscriber_id": [f"S{i:03d}" for i in range(n)],
        "activation_date": pd.date_range("2025-01-01", periods=n, freq="3D").strftime("%Y-%m-%d"),
        "plan": ["Prepaid", "Postpaid"] * 20,
        "minutes_used": [str(100 + i) for i in range(n)],
        "data_mb": [float(500 + 10 * i) for i in range(n)],
    })
    df.loc[8, "minutes_used"] = " "      # blanc caché
    df.loc[5, "plan"] = "postpaid"       # variante de casse
    df.loc[12, "data_mb"] = -50.0        # valeur impossible
    return pd.concat([df, df.iloc[[7]]], ignore_index=True)


def test_usage_dataset_dates_survive_and_negative_values_are_reported(tmp_path):
    report, frames = run_with_data(_request(_write(tmp_path, _usage_frame()), tmp_path))
    df = frames["t"]
    assert pd.api.types.is_datetime64_any_dtype(df["activation_date"])  # plus de hachage
    assert pd.api.types.is_numeric_dtype(df["minutes_used"]) and df["minutes_used"].isna().sum() == 0
    assert set(df["plan"]) == {"Prepaid", "Postpaid"} and len(df) == 40
    negative = [c for c in report.validation_checks if c.check == "range" and not c.passed]
    assert [c.column for c in negative] == ["data_mb"] and negative[0].violations == 1
    assert "automatique" in negative[0].detail
    assert (df["data_mb"] < 0).sum() == 1  # signalée, jamais corrigée en silence
    assert "dim_date" in {d.name for d in report.star_schemas[0].dimensions} or \
        report.star_schemas[0].date_columns == ["activation_date"]


def _write(tmp_path, df):
    path = tmp_path / "usage.csv"
    df.to_csv(path, index=False)
    return path

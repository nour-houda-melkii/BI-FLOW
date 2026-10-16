"""Tests du modèle en étoile : proposition, construction, contrôles sans perte."""

import numpy as np
import pandas as pd
import pytest

from agents.etl import ETLOptions, ETLRequest, run_with_data
from agents.etl.models import StarSchemaSpec
from agents.profiler.models import ProfilingReport, SourceConfig, TableProfile
from agents.tools import modeling as M


def _telecom_frame(n=60) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "city": rng.choice(["Tunis", "Sfax", "Sousse"], n),
        "contract": rng.choice(["Monthly", "Yearly"], n),
        "payment_method": rng.choice(["Card", "Cash"], n),
        "internet_service": rng.choice(["DSL", "Fiber", "No"], n),
        "monthly_charges": rng.uniform(10, 100, n).round(2),
        "data_usage_gb": rng.uniform(0, 50, n).round(1),
        "churn_value": rng.choice([0, 1], n),
        "const_col": "x",
    })


def _plan(df, **kw):
    roles = {c: ("id" if c == "customer_id" else "measure" if M.is_numeric(df[c]) else "dimension")
             for c in df.columns}
    return M.propose_star_schema(df, roles, table="subs", **kw)


# ---------------------------------------------------------------- proposition
def test_grain_is_the_unique_identifier():
    plan = _plan(_telecom_frame())
    assert plan["grain_key"] == "customer_id" and plan["fact_name"] == "fact_customer"
    assert "1 customer" in plan["grain"]


def test_measures_dimensions_and_exclusions_are_classified():
    plan = _plan(_telecom_frame())
    measures = {m["column"]: m for m in plan["measures"]}
    assert set(measures) == {"monthly_charges", "data_usage_gb", "churn_value"}
    assert measures["churn_value"]["kind"] == "flag"
    assert measures["monthly_charges"]["suggested_aggregations"] == ["sum", "avg"]
    dims = {d["name"]: d["columns"] for d in plan["dimensions"]}
    assert dims["dim_location"] == ["city"]
    assert dims["dim_contract"] == ["contract", "payment_method"]
    assert dims["dim_service"] == ["internet_service"]
    assert dims["dim_customer"] == ["customer_id", "gender"]
    assert [e["column"] for e in plan["excluded"]] == ["const_col"]


def test_geo_numeric_columns_are_not_measures():
    df = _telecom_frame().assign(latitude=np.linspace(30, 40, 60))
    plan = _plan(df)
    assert "latitude" not in {m["column"] for m in plan["measures"]}
    assert "latitude" in next(d for d in plan["dimensions"] if d["name"] == "dim_location")["columns"]


def test_without_unique_identifier_the_grain_is_the_row():
    df = _telecom_frame().drop(columns="customer_id")
    plan = _plan(df)
    assert plan["grain_key"] is None and "ligne" in plan["grain"]
    fact, dims = M.build_star_tables(df, plan)
    assert fact["fact_row_key"].is_unique
    assert all(c["passed"] for c in M.check_star(df, fact, dims, plan))


def test_manual_spec_overrides_the_automatic_proposal():
    spec = StarSchemaSpec(
        fact_name="fact_usage", measures=["monthly_charges"],
        dimensions={"dim_plan": ["contract"], "dim_geo": ["city"]}, exclude=["data_usage_gb"])
    plan = _plan(_telecom_frame(), spec=spec)
    assert plan["source"] == "configured" and plan["fact_name"] == "fact_usage"
    assert [m["column"] for m in plan["measures"]] == ["monthly_charges"]
    assert {d["name"] for d in plan["dimensions"]} == {"dim_plan", "dim_geo", "dim_customer"}


# ---------------------------------------------------------------- construction
def test_star_is_lossless_and_all_checks_pass():
    df = _telecom_frame()
    plan = _plan(df)
    fact, dims = M.build_star_tables(df, plan)
    checks = M.check_star(df, fact, dims, plan)
    assert all(c["passed"] for c in checks), [c for c in checks if not c["passed"]]
    assert len(fact) == len(df)
    assert fact["monthly_charges"].sum() == pytest.approx(df["monthly_charges"].sum())


def test_dimensions_are_deduplicated_and_keys_start_at_one():
    df = _telecom_frame()
    plan = _plan(df)
    _, dims = M.build_star_tables(df, plan)
    contract = dims["dim_contract"]
    assert len(contract) <= 4 and contract["contract_key"].min() == 1
    assert contract["contract_key"].is_unique


def test_surrogate_keys_are_deterministic():
    df = _telecom_frame()
    plan = _plan(df)
    f1, _ = M.build_star_tables(df, plan)
    f2, _ = M.build_star_tables(df.sample(frac=1, random_state=3).sort_index(), plan)
    pd.testing.assert_frame_equal(f1, f2)


def test_nulls_in_dimension_columns_are_handled():
    df = _telecom_frame()
    df.loc[0, "contract"] = None
    plan = _plan(df)
    fact, dims = M.build_star_tables(df, plan)
    assert all(c["passed"] for c in M.check_star(df, fact, dims, plan))


def test_date_dimension_is_built_for_datetime_columns():
    df = _telecom_frame().assign(
        activation_date=pd.date_range("2024-01-30", periods=60, freq="D"))
    plan = _plan(df)
    fact, dims = M.build_star_tables(df, plan)
    assert plan["date_columns"] == ["activation_date"]
    assert "activation_date_date_key" in fact.columns
    assert fact["activation_date_date_key"].iloc[0] == 20240130
    assert {"year", "quarter", "month", "weekday"} <= set(dims["dim_date"].columns)


def test_mermaid_diagram_lists_fact_and_dimensions():
    df = _telecom_frame()
    plan = _plan(df)
    fact, dims = M.build_star_tables(df, plan)
    text = M.to_mermaid(plan, fact, dims)
    assert text.startswith("erDiagram") and "fact_customer" in text
    assert "dim_contract ||--o{ fact_customer : contract_key" in text


# ---------------------------------------------------------------- les contrôles détectent les erreurs
def _built():
    df = _telecom_frame()
    plan = _plan(df)
    fact, dims = M.build_star_tables(df, plan)
    return df, plan, fact, dims


def _failed(checks):
    return {c["check"] for c in checks if not c["passed"]}


def test_check_detects_orphan_foreign_key():
    df, plan, fact, dims = _built()
    fact.loc[0, "contract_key"] = 999
    assert "star_no_orphan_foreign_key" in _failed(M.check_star(df, fact, dims, plan))


def test_check_detects_corrupted_dimension_value():
    df, plan, fact, dims = _built()
    dims["dim_contract"].loc[0, "contract"] = "CORRUPTED"
    assert "star_lossless_rejoin" in _failed(M.check_star(df, fact, dims, plan))


def test_check_detects_lost_rows_and_wrong_measures():
    df, plan, fact, dims = _built()
    assert "star_fact_row_count" in _failed(M.check_star(df, fact.iloc[:-1], dims, plan))
    fact["monthly_charges"] = fact["monthly_charges"] * 2
    assert "star_measure_reconciliation" in _failed(M.check_star(df, fact, dims, plan))


def test_check_detects_a_column_left_out():
    df, plan, fact, dims = _built()
    plan["dimensions"] = [d for d in plan["dimensions"] if d["name"] != "dim_service"]
    assert "star_column_coverage" in _failed(M.check_star(df, fact, dims, plan))


# ---------------------------------------------------------------- dans l'agent
def _run(df, tmp_path, **opts):
    csv = tmp_path / "subs.csv"
    df.to_csv(csv, index=False)
    request = ETLRequest(
        profiling_report=ProfilingReport(
            business_goal="churn",
            tables=[TableProfile(name="subs", source_name="subs", row_count=0, column_count=0)]),
        sources=[SourceConfig(name="subs", type="csv", uri=str(csv))],
        options=ETLOptions(output_dir=str(tmp_path / "out"), use_llm=False, **opts))
    return run_with_data(request)


def test_agent_writes_star_files_and_describes_them_in_the_report(tmp_path):
    report, _ = _run(_telecom_frame(), tmp_path)
    star = report.star_schemas[0]
    assert star.fact_name == "fact_customer" and star.fact_rows == 60
    model_dir = tmp_path / "out" / "model" / "subs"
    for name in ("fact_customer", "dim_customer", "dim_contract", "dim_location", "dim_service"):
        assert (model_dir / f"{name}.csv").exists()
    assert (model_dir / "star_schema.mmd").read_text(encoding="utf-8").startswith("erDiagram")
    assert {"star_lossless_rejoin", "star_no_orphan_foreign_key"} <= {
        c.check for c in report.validation_checks}
    assert all(c.passed for c in report.validation_checks if c.check.startswith("star"))


def test_saved_csv_files_rejoin_to_the_clean_table(tmp_path):
    """Test de bout en bout sur les fichiers écrits, comme le ferait l'agent KPI."""
    report, frames = _run(_telecom_frame(), tmp_path)
    d = tmp_path / "out" / "model" / "subs"
    fact = pd.read_csv(d / "fact_customer.csv")
    for dim_name, key in [("dim_customer", "customer_key"), ("dim_contract", "contract_key"),
                          ("dim_location", "location_key"), ("dim_service", "service_key")]:
        fact = fact.merge(pd.read_csv(d / f"{dim_name}.csv"), on=key, how="left", validate="m:1")
    clean = frames["subs"].drop(columns="const_col")
    assert len(fact) == len(clean)
    assert fact["monthly_charges"].sum() == pytest.approx(clean["monthly_charges"].sum())
    assert fact.groupby("contract")["churn_value"].mean().round(6).to_dict() == \
        clean.groupby("contract")["churn_value"].mean().round(6).to_dict()


def test_star_can_be_disabled(tmp_path):
    report, _ = _run(_telecom_frame(), tmp_path, build_star_schema=False)
    assert report.star_schemas == [] and not (tmp_path / "out" / "model").exists()


def test_table_without_measures_is_skipped_without_error(tmp_path):
    df = pd.DataFrame({"id": [f"A{i}" for i in range(20)], "color": ["r", "g"] * 10})
    report, _ = _run(df, tmp_path)
    assert report.star_schemas == [] and not report.errors
    assert any(i.rule == "star_schema_skipped" for i in report.remaining_issues)


def test_telco_star_schema_end_to_end(tmp_path):
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    csv = root / "data" / "telco.csv"
    if not csv.exists():
        pytest.skip("data/telco.csv absent")
    profile = ProfilingReport.model_validate_json(
        (Path(__file__).parent / "fixtures" / "profiler_output.json").read_text(encoding="utf-8"))
    report, _ = run_with_data(ETLRequest(
        profiling_report=profile, sources=[SourceConfig(name="telco", type="csv", uri=str(csv))],
        options=ETLOptions(preset="telco", output_dir=str(tmp_path), use_llm=False)))
    star = report.star_schemas[0]
    assert report.status == "success"
    assert star.fact_name == "fact_customer" and star.fact_rows == 7043
    assert {d.name for d in star.dimensions} == {
        "dim_customer", "dim_churn", "dim_contract", "dim_location", "dim_service"}
    assert {m.column for m in star.measures} == {
        "tenure_months", "monthly_charges", "total_charges", "churn_value", "churn_score", "cltv"}
    assert {e["column"] for e in star.excluded_columns} == {"count", "country", "state"}
    assert len(star.check_ids) == 7


def test_pseudonymised_personal_columns_go_to_the_entity_dimension():
    df = _telecom_frame().assign(email=[f"h{i:016d}" for i in range(60)])
    roles = {c: ("id" if c == "customer_id" else "dimension") for c in df.columns}
    plan = M.propose_star_schema(df, roles, table="subs", entity_columns={"email"})
    entity = next(d for d in plan["dimensions"] if d["role"] == "entity")
    assert "email" in entity["columns"]
    assert "dim_attributes" not in {d["name"] for d in plan["dimensions"]}

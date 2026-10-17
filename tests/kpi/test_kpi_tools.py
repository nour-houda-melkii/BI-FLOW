"""Tests du calcul déterministe des KPI (pandas uniquement, sans LLM ni LangGraph)."""

import pandas as pd
import pytest

from agents.kpi.catalog import CATALOG
from agents.tools.kpi import as_flag, compute_breakdown, compute_value, pick_dimensions, resolve_inputs

KPI = {k.id: k for k in CATALOG}


@pytest.fixture
def df():
    rows = 100
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(rows)],
        "monthly_charges": [50.0] * 60 + [100.0] * 40,
        "churn_label": ["Yes"] * 25 + ["No"] * 75,
        "contract": ["Month-to-month"] * 50 + ["Two year"] * 50,
    })


def _value(df, kpi_id):
    cols, missing = resolve_inputs(KPI[kpi_id], df)
    assert not missing
    return compute_value(KPI[kpi_id], df, cols)


def test_churn_rate_from_text_flag(df):
    assert _value(df, "churn_rate") == 25.0


def test_arpu_and_revenue_lost(df):
    assert _value(df, "arpu") == 70.0
    expected = df.loc[df.churn_label == "Yes", "monthly_charges"].sum()
    assert _value(df, "monthly_revenue_lost") == expected


def test_missing_columns_are_reported_not_guessed(df):
    _, missing = resolve_inputs(KPI["dropped_call_rate"], df)
    assert set(missing) == {"num", "den"}


def test_undefined_values_return_none(df):
    cols, _ = resolve_inputs(KPI["churn_rate"], df)
    assert compute_value(KPI["churn_rate"], df.iloc[0:0], cols) is None


def test_non_binary_column_is_not_a_flag():
    assert as_flag(pd.Series(["Maybe", "Yes"])) is None


def test_breakdown_skips_small_segments_and_pii(df):
    meta = {"columns": [
        {"name": "contract", "semantic_role": "dimension"},
        {"name": "customer_id", "semantic_role": "dimension", "is_pii": True},
    ]}
    dims = pick_dimensions(meta, df, max_segments=10)
    assert dims == ["contract"]
    cols, _ = resolve_inputs(KPI["churn_rate"], df)
    assert compute_breakdown(KPI["churn_rate"], df, cols, dims, min_rows=60) == {}
    assert len(compute_breakdown(KPI["churn_rate"], df, cols, dims, min_rows=30)["contract"]) == 2


def test_churn_kpi_is_not_broken_down_by_a_churn_column(df):
    df = df.assign(churn_value=(df.churn_label == "Yes").astype(int))
    cols, _ = resolve_inputs(KPI["churn_rate"], df)
    assert cols["flag"] == "churn_value"
    assert compute_breakdown(KPI["churn_rate"], df, cols, ["churn_label"], min_rows=1) == {}
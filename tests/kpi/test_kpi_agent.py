"""Test de bout en bout du KPI Agent (rapport ETL minimal + DataFrame en mémoire, sans LLM)."""

import pandas as pd

from agents.kpi import KPIOptions, KPIRequest, run


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(100)],
        "monthly_charges": [50.0] * 60 + [100.0] * 40,
        "churn_label": ["Yes"] * 25 + ["No"] * 75,
        "contract": ["Month-to-month"] * 50 + ["Two year"] * 50,
    })


def _etl_report() -> dict:
    return {
        "business_goal": "Analyse BI telecom", "status": "success",
        "tables": [{
            "name": "telco", "rows_out": 100, "output_path": None,
            "quality_after": {"overall": 99.0},
            "columns": [{"name": "contract", "semantic_role": "dimension",
                         "is_pii": False, "pii_treatment": None}],
        }],
        "remaining_issues": [],
    }


def test_agent_end_to_end(tmp_path):
    request = KPIRequest(etl_report=_etl_report(),
                         options=KPIOptions(output_dir=str(tmp_path), use_llm=False))
    report = run(request, frames={"telco": _frame()})

    assert report.status == "success", report.errors
    by_id = {k.id: k for k in report.kpis}
    assert by_id["churn_rate"].status == "computed" and by_id["churn_rate"].value == 25.0
    assert by_id["arpu"].value == 70.0
    assert "contract" in by_id["arpu"].breakdowns
    assert by_id["dropped_call_rate"].status == "not_computable"
    assert (tmp_path / "kpi_report.json").exists()
    assert (tmp_path / "kpi_values.csv").exists()
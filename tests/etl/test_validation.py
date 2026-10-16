"""Tests des scores qualité et des contrôles de validation."""

import pandas as pd

from agents.etl.models import MappingRule, ProductRule, RangeRule
from agents.tools import validation as V

NULL_TOKENS = ["", "n/a", "na", "null", "nan", "#n/a"]


def _snap(df, **kw):
    base = dict(null_tokens=NULL_TOKENS, structural={}, range_rules=[], mapping_rules=[])
    base.update(kw)
    return V.quality_snapshot(df, **base)


def test_perfect_frame_scores_100():
    s = _snap(pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
    assert s["overall"] == 100.0 and s["null_cells"] == 0


def test_blank_strings_count_as_missing():
    s = _snap(pd.DataFrame({"a": ["x", " ", "y", "z"]}))
    assert s["null_cells"] == 1 and s["completeness"] < 100


def test_numeric_stored_as_text_lowers_validity():
    s = _snap(pd.DataFrame({"amount": ["1.5", "2.5", "3.5", "4.5"], "n": [1, 2, 3, 4]}),
              range_rules=[RangeRule(column="n", min=0)])
    assert s["text_numeric_columns"] == ["amount"] and s["validity"] < 100


def test_structural_nulls_do_not_hurt_completeness():
    df = pd.DataFrame({"label": ["Yes"] * 5 + ["No"] * 5, "reason": ["r"] * 5 + [None] * 5})
    s = _snap(df, structural={"reason": {"driver": "label", "value": "No", "null_count": 5}})
    assert s["completeness"] == 100.0


def test_duplicates_and_mapping_violations_are_penalised():
    df = pd.DataFrame({"lab": ["Yes", "No", "No", "No"], "val": [1, 1, 0, 0]})
    df = pd.concat([df, df.iloc[[2]]])
    rule = MappingRule(column_a="lab", column_b="val", mapping={"Yes": 1, "No": 0})
    s = _snap(df, mapping_rules=[rule])
    assert s["uniqueness"] < 100 and s["consistency"] < 100


def _validate(df, **kw):
    base = dict(key=[], structural={}, expected_numeric=[], range_rules=[], mapping_rules=[],
                product_rules=[], rows_in=len(df), rows_removed=0)
    base.update(kw)
    return {c["check"] + (f".{c['column']}" if c["column"] else ""): c
            for c in V.validate_clean_table("t", df, **base)}


def test_row_reconciliation_detects_silent_row_loss():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert _validate(df, rows_in=3)["row_reconciliation"]["passed"]
    lost = _validate(df, rows_in=5)["row_reconciliation"]
    assert not lost["passed"] and lost["severity"] == "high" and lost["violations"] == 2


def test_key_checks():
    df = pd.DataFrame({"id": [1, 1, None], "v": [1, 2, 3]})
    checks = _validate(df, key=["id"])
    assert not checks["business_key_unique"]["passed"]
    assert not checks["business_key_not_null"]["passed"]


def test_expected_numeric_dtype_check():
    df = pd.DataFrame({"amount": ["1", "2"]})
    assert not _validate(df, expected_numeric=["amount"])["numeric_dtype.amount"]["passed"]


def test_range_mapping_and_product_checks():
    df = pd.DataFrame({
        "age": [10, -1, 5], "lab": ["Yes", "No", "No"], "val": [1, 1, 0],
        "tenure": [2, 2, 2], "monthly": [10.0, 10.0, 10.0], "total": [20.0, 20.0, 90.0],
    })
    checks = _validate(
        df,
        range_rules=[RangeRule(column="age", min=0)],
        mapping_rules=[MappingRule(column_a="lab", column_b="val", mapping={"Yes": 1, "No": 0})],
        product_rules=[ProductRule(column="total", factor_a="tenure", factor_b="monthly",
                                   tolerance_pct=25)],
    )
    assert checks["range.age"]["violations"] == 1
    assert checks["mapping_consistency.lab->val"]["violations"] == 1
    assert checks["product_plausibility.total"]["violations"] == 1
    assert checks["product_plausibility.total"]["severity"] == "low"


def test_validity_is_not_zero_when_only_one_column_is_mistyped():
    df = pd.DataFrame({"minutes": ["1", "2", "3", "4"], "data_mb": [1.0, 2.0, 3.0, 4.0]})
    s = _snap(df)
    assert s["text_numeric_columns"] == ["minutes"] and s["validity"] == 50.0

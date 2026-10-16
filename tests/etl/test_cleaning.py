"""Tests unitaires des outils de nettoyage (aucun LLM, aucun fichier)."""

import numpy as np
import pandas as pd
import pytest

from agents.etl.models import ConditionalFillRule, MappingRule, RangeRule
from agents.tools import cleaning as C

NULL_TOKENS = ["", "n/a", "na", "null", "nan", "#n/a"]


# ---------------------------------------------------------------- texte
def test_normalize_turns_blanks_into_nulls_without_mutating_input():
    df = pd.DataFrame({"a": ["x", " ", "N/A", "  y  ", None]})
    original = df.copy(deep=True)

    out, actions = C.normalize_text_columns(df, null_tokens=NULL_TOKENS)

    pd.testing.assert_frame_equal(df, original)  # l'entrée n'est pas modifiée
    assert out["a"].isna().sum() == 3  # ' ', 'N/A' et le None d'origine
    assert out["a"].dropna().tolist() == ["x", "y"]
    rules = {a["rule"]: a["rows_affected"] for a in actions}
    assert rules["blank_to_null"] == 2
    assert rules["strip_whitespace"] == 1


def test_normalize_leaves_numeric_columns_alone():
    df = pd.DataFrame({"n": [1, 2, 3]})
    out, actions = C.normalize_text_columns(df, null_tokens=NULL_TOKENS)
    assert actions == []
    assert out["n"].tolist() == [1, 2, 3]


# ---------------------------------------------------------------- types
def test_coerce_numeric_converts_text_amounts():
    df = pd.DataFrame({"amount": ["10.5", "1,234.50", "$20", None, "7"]})
    out, actions = C.coerce_numeric_columns(df, min_ratio=0.9, protected=set())
    assert pd.api.types.is_float_dtype(out["amount"])
    assert out["amount"].tolist()[:3] == [10.5, 1234.5, 20.0]
    assert out["amount"].isna().sum() == 1
    assert actions[0]["rule"] == "coerce_to_numeric"


def test_coerce_numeric_protects_identifier_with_leading_zeros():
    df = pd.DataFrame({"zip_code": ["01234", "00501", "90210"]})
    out, actions = C.coerce_numeric_columns(df, min_ratio=0.9, protected=set())
    assert out["zip_code"].tolist() == ["01234", "00501", "90210"]  # zéros conservés
    assert actions == []


def test_coerce_numeric_skips_protected_columns():
    df = pd.DataFrame({"customer_id": ["1001", "1002", "1003"]})
    out, _ = C.coerce_numeric_columns(df, min_ratio=0.9, protected={"customer_id"})
    assert out["customer_id"].tolist() == ["1001", "1002", "1003"]


def test_coerce_numeric_reports_unparseable_values_as_nulls():
    df = pd.DataFrame({"v": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "oops"]})
    out, actions = C.coerce_numeric_columns(df, min_ratio=0.9, protected=set())
    assert out["v"].isna().sum() == 1
    assert "unparseable_to_null" in {a["rule"] for a in actions}


def test_coerce_numeric_ignores_genuine_text():
    df = pd.DataFrame({"city": ["Paris", "Tunis", "Lyon", "Tunis"]})
    out, actions = C.coerce_numeric_columns(df, min_ratio=0.9, protected=set())
    assert actions == [] and out["city"].tolist() == df["city"].tolist()


def test_harmonize_case_merges_variants_to_most_frequent():
    df = pd.DataFrame({"flag": ["Yes", "Yes", "Yes", "yes", "YES", "No"]})
    out, actions = C.harmonize_categories(df, protected=set())
    assert set(out["flag"].dropna()) == {"Yes", "No"}
    assert actions[0]["rows_affected"] == 2


# ---------------------------------------------------------------- manquants
def _structural_frame(n=40):
    label = ["Yes" if i % 4 == 0 else "No" for i in range(n)]
    reason = [f"r{i % 3}" if lab == "Yes" else None for i, lab in enumerate(label)]
    return pd.DataFrame({"label": label, "reason": reason, "x": range(n)})


def test_detect_structural_missing_finds_driver():
    found = C.detect_structural_missing(_structural_frame())
    assert found["reason"]["driver"] == "label"
    assert found["reason"]["value"] == "No"


def test_detect_structural_missing_ignores_random_nulls():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "g": rng.choice(["a", "b"], 200),
        "v": rng.choice(["p", "q", None], 200),
    })
    assert C.detect_structural_missing(df) == {}


def test_conditional_fill_only_touches_matching_rows():
    df = pd.DataFrame({"tenure": [0, 0, 5], "total": [None, 3.0, None]})
    rule = ConditionalFillRule(column="total", value=0.0, when_column="tenure", when_equals=0)
    out, actions = C.apply_conditional_fills(df, [rule])
    assert out["total"].tolist()[:2] == [0.0, 3.0]
    assert pd.isna(out["total"].iloc[2])  # tenure=5 non touché
    assert actions[0]["rows_affected"] == 1


def _missing_kwargs(**over):
    base = dict(
        structural={}, protected=set(), numeric_strategy="median",
        categorical_strategy="unknown", unknown_label="Unknown",
        not_applicable_label="Not applicable", drop_threshold_pct=80.0,
    )
    base.update(over)
    return base


def test_missing_numeric_uses_median_and_explains_why():
    df = pd.DataFrame({"n": [1.0, 2.0, 100.0, None]})
    out, actions = C.handle_missing_values(df, **_missing_kwargs())
    assert out["n"].iloc[3] == 2.0
    assert actions[0]["rule"] == "impute_median" and actions[0]["rationale"]


def test_missing_categorical_gets_unknown_label():
    df = pd.DataFrame({"c": ["a", "b", None, "a"]})
    out, _ = C.handle_missing_values(df, **_missing_kwargs())
    assert out["c"].iloc[2] == "Unknown"


def test_missing_never_imputes_protected_identifiers():
    df = pd.DataFrame({"customer_id": ["a", None, "c"]})
    out, actions = C.handle_missing_values(df, **_missing_kwargs(protected={"customer_id"}))
    assert out["customer_id"].isna().sum() == 1 and actions == []


def test_missing_structural_text_is_labelled_not_applicable():
    df = _structural_frame()
    structural = C.detect_structural_missing(df)
    out, actions = C.handle_missing_values(df, **_missing_kwargs(structural=structural))
    assert out["reason"].isna().sum() == 0
    assert (out["reason"] == "Not applicable").sum() == 30
    assert actions[0]["rule"] == "structural_missing_fill"


def test_missing_sparse_column_is_dropped():
    df = pd.DataFrame({"keep": range(10), "sparse": [1.0] + [None] * 9})
    out, actions = C.handle_missing_values(df, **_missing_kwargs())
    assert "sparse" not in out.columns
    assert actions[0]["rule"] == "drop_sparse_column"


# ---------------------------------------------------------------- doublons
def test_drop_exact_duplicates_keeps_first():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    out, actions = C.drop_exact_duplicates(df, table="t")
    assert len(out) == 2 and actions[0]["rows_affected"] == 1
    assert actions[0]["source_issue_id"] == "t.exact_duplicates"


def test_key_duplicates_flag_strategy_removes_nothing():
    df = pd.DataFrame({"id": [1, 1, 2], "v": ["a", "b", "c"]})
    out, actions = C.resolve_key_duplicates(df, key=["id"], strategy="flag", table="t")
    assert len(out) == 3 and actions[0]["rule"] == "flag_key_duplicates"


@pytest.mark.parametrize("strategy,expected", [("keep_first", "a"), ("keep_last", "b")])
def test_key_duplicates_keep_strategies(strategy, expected):
    df = pd.DataFrame({"id": [1, 1, 2], "v": ["a", "b", "c"]})
    out, _ = C.resolve_key_duplicates(df, key=["id"], strategy=strategy, table="t")
    assert len(out) == 2 and out.loc[out["id"] == 1, "v"].iloc[0] == expected


# ---------------------------------------------------------------- outliers
def _outlier_frame():
    return pd.DataFrame({"v": [10, 11, 12, 10, 11, 12, 10, 11, 12, 10, 11, 12, 10, 11, 1000]})


def test_outliers_are_reported_but_not_modified_by_default():
    df = _outlier_frame()
    out, actions, counts = C.handle_outliers(
        df, protected=set(), strategy="report", iqr_multiplier=1.5)
    pd.testing.assert_frame_equal(out, df)
    assert counts == {"v": 1} and actions[0]["rule"] == "report_outliers"


def test_outliers_clip_strategy_bounds_values():
    out, actions, _ = C.handle_outliers(
        _outlier_frame(), protected=set(), strategy="clip", iqr_multiplier=1.5)
    assert out["v"].max() < 1000 and actions[0]["rule"] == "clip_outliers"


def test_outliers_skip_binary_columns():
    df = pd.DataFrame({"flag": [0] * 50 + [1]})
    _, _, counts = C.handle_outliers(df, protected=set(), strategy="report", iqr_multiplier=1.5)
    assert counts == {}


# ---------------------------------------------------------------- PII
def test_pii_hash_is_deterministic_and_salt_dependent():
    df = pd.DataFrame({"email": [f"user{i}@mail.com" for i in range(30)]})
    a, _, t = C.apply_pii_policy(df, {"email": ["email"]}, pii_policy="hash",
                                 identifier_policy="keep", salt="s1")
    b, _, _ = C.apply_pii_policy(df, {"email": ["email"]}, pii_policy="hash",
                                 identifier_policy="keep", salt="s1")
    c, _, _ = C.apply_pii_policy(df, {"email": ["email"]}, pii_policy="hash",
                                 identifier_policy="keep", salt="s2")
    assert t["email"] == "hashed"
    assert a["email"].tolist() == b["email"].tolist()
    assert a["email"].tolist() != c["email"].tolist()
    assert "@" not in "".join(a["email"])
    assert a["email"].nunique() == 30  # l'unicité est préservée


def test_pii_drop_policy_removes_column():
    df = pd.DataFrame({"email": [f"u{i}@x.com" for i in range(30)], "v": range(30)})
    out, _, t = C.apply_pii_policy(df, {"email": ["email"]}, pii_policy="drop",
                                   identifier_policy="keep", salt="s")
    assert "email" not in out.columns and t["email"] == "dropped"


def test_pii_identifier_kept_by_default():
    df = pd.DataFrame({"customer_id": [f"C{i}" for i in range(30)]})
    out, _, t = C.apply_pii_policy(df, {"customer_id": ["customer_identifier"]},
                                   pii_policy="hash", identifier_policy="keep", salt="s")
    assert out["customer_id"].tolist() == df["customer_id"].tolist() and t["customer_id"] == "kept"


def test_pii_false_positives_are_ignored():
    df = pd.DataFrame({
        "lat_long": [f"{30 + i / 10:.4f}, -{100 + i / 10:.4f}" for i in range(50)],
        "phone_service": ["Yes", "No"] * 25,
        "latitude": np.linspace(30, 40, 50),
    })
    out, actions, t = C.apply_pii_policy(
        df, {c: ["phone"] for c in df.columns}, pii_policy="hash",
        identifier_policy="keep", salt="s")
    pd.testing.assert_frame_equal(out, df)
    assert set(t.values()) == {"false_positive_ignored"}


# ---------------------------------------------------------------- noms / règles
def test_snake_case_names_and_collisions():
    m = C.snake_case_mapping(["Total Charges", "CustomerID", "Zip Code", "total_charges"])
    assert m["Total Charges"] == "total_charges"
    assert m["CustomerID"] == "customer_id"
    assert m["Zip Code"] == "zip_code"
    assert m["total_charges"] == "total_charges_2"


def test_looks_like_identifier():
    assert C.looks_like_identifier("CustomerID")
    assert C.looks_like_identifier("Zip Code")
    assert not C.looks_like_identifier("Paid")
    assert not C.looks_like_identifier("Monthly Charges")


def test_range_rule_set_null_and_mapping_derive():
    df = pd.DataFrame({"age": [10, -5, 30], "lab": ["Yes", "No", "Yes"], "val": [1, 1, 1]})
    out, _ = C.apply_range_rules(df, [RangeRule(column="age", min=0, fix="set_null")])
    assert pd.isna(out["age"].iloc[1])
    out, actions = C.apply_mapping_rules(
        df, [MappingRule(column_a="lab", column_b="val", mapping={"Yes": 1, "No": 0},
                         fix="derive_b")])
    assert out["val"].tolist() == [1, 0, 1] and actions[0]["rows_affected"] == 1


# ---------------------------------------------------------------- régressions (2e dataset)
def test_pii_guard_ignores_date_columns_flagged_as_phone():
    dates = pd.DataFrame({"activation_date": pd.date_range("2025-01-01", periods=40)})
    out, _, t = C.apply_pii_policy(dates, {"activation_date": ["phone"]}, pii_policy="hash",
                                   identifier_policy="keep", salt="s")
    assert t["activation_date"] == "false_positive_ignored"
    assert pd.api.types.is_datetime64_any_dtype(out["activation_date"])

    text_dates = pd.DataFrame({"d": [f"2025-01-{i % 28 + 1:02d}" for i in range(40)]})
    out, _, t = C.apply_pii_policy(text_dates, {"d": ["phone"]}, pii_policy="hash",
                                   identifier_policy="keep", salt="s")
    assert t["d"] == "false_positive_ignored" and out["d"].tolist() == text_dates["d"].tolist()


def test_real_phone_numbers_are_still_hashed():
    df = pd.DataFrame({"msisdn": [f"+216 20 {i:03d} 456" for i in range(40)]})
    out, _, t = C.apply_pii_policy(df, {"msisdn": ["phone"]}, pii_policy="hash",
                                   identifier_policy="keep", salt="s")
    assert t["msisdn"] == "hashed" and not out["msisdn"].str.contains("216").any()


@pytest.mark.parametrize("name,expected", [
    ("data_mb", True), ("Minutes Used", True), ("Monthly Charges", True), ("tenure_months", True),
    ("account_balance", False), ("latitude", False), ("plan", False), ("churn_score", False),
])
def test_non_negative_name_heuristic(name, expected):
    assert C.is_non_negative_name(name) is expected

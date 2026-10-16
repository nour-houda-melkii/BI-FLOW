"""Scores qualité et contrôles de validation de l'ETL / Data Quality Agent."""

from __future__ import annotations

from typing import Any

import pandas as pd

from agents.tools.cleaning import (
    is_numeric,
    is_text,
    normalize_text_columns,
    numeric_like_columns,
)


def count_range_violations(frame: pd.DataFrame, rule: Any) -> int:
    if rule.column not in frame.columns:
        return 0
    values = pd.to_numeric(frame[rule.column], errors="coerce")
    bad = pd.Series(False, index=frame.index)
    if rule.min is not None:
        bad |= values < rule.min
    if rule.max is not None:
        bad |= values > rule.max
    return int(bad.sum())


def count_mapping_violations(frame: pd.DataFrame, rule: Any) -> int:
    if rule.column_a not in frame.columns or rule.column_b not in frame.columns:
        return 0
    a = frame[rule.column_a].astype("string").str.strip()
    b = frame[rule.column_b].astype("string").str.strip()
    total = 0
    for a_value, b_value in rule.mapping.items():
        rows = a == str(a_value)
        total += int((rows & (b != str(b_value))).fillna(False).sum())
    return total


def count_product_violations(frame: pd.DataFrame, rule: Any) -> int:
    needed = {rule.column, rule.factor_a, rule.factor_b}
    if not needed <= set(frame.columns):
        return 0
    value = pd.to_numeric(frame[rule.column], errors="coerce")
    expected = (
        pd.to_numeric(frame[rule.factor_a], errors="coerce")
        * pd.to_numeric(frame[rule.factor_b], errors="coerce")
    )
    valid = value.notna() & expected.notna() & (expected > 0)
    deviation = ((value - expected).abs() / expected.where(expected > 0)) * 100.0
    return int((valid & (deviation > rule.tolerance_pct)).sum())


def _structural_null_cells(frame: pd.DataFrame, structural: dict[str, dict[str, Any]]) -> int:
    return sum(int(frame[c].isna().sum()) for c in structural if c in frame.columns)


def quality_snapshot(
    frame: pd.DataFrame,
    *,
    null_tokens: list[str],
    structural: dict[str, dict[str, Any]],
    range_rules: list[Any],
    mapping_rules: list[Any],
    min_ratio: float = 0.9,
    protected: set[str] | None = None,
) -> dict[str, Any]:
    """Mesure déterministe sur 4 dimensions. Les blancs comptent comme manquants.

    Les nulls structurels (ex. motif de départ d'un client resté) sont exclus de la
    complétude : ils ne sont pas un défaut de qualité.
    """
    rows, cols = len(frame), len(frame.columns)
    if rows == 0 or cols == 0:
        return {
            "completeness": 100.0, "uniqueness": 100.0, "validity": 100.0,
            "consistency": 100.0, "overall": 100.0, "rows": rows,
            "null_cells": 0, "text_numeric_columns": [],
        }

    normalized, _ = normalize_text_columns(frame, null_tokens=null_tokens)

    # Complétude
    null_cells = int(normalized.isna().sum().sum())
    structural_cells = _structural_null_cells(normalized, structural)
    effective_nulls = max(null_cells - structural_cells, 0)
    completeness = (1 - effective_nulls / (rows * cols)) * 100.0

    # Unicité
    duplicates = int(normalized.duplicated().sum())
    uniqueness = (1 - duplicates / rows) * 100.0

    # Validité : une colonne numérique stockée en texte est inutilisable pour un calcul
    # (somme, moyenne) -> toutes ses cellules comptent comme invalides ; s'ajoutent les
    # valeurs hors domaine des règles de plage.
    text_numeric = numeric_like_columns(normalized, min_ratio=min_ratio, protected=protected)
    numeric_dtype = {str(c) for c in normalized.columns if is_numeric(normalized[c])}
    checked = set(text_numeric) | numeric_dtype  # toutes les colonnes censées être numériques
    invalid = rows * len(text_numeric)
    for rule in range_rules:
        if rule.column in normalized.columns:
            checked.add(rule.column)
            if rule.column not in text_numeric:
                invalid += count_range_violations(normalized, rule)
    validity = (1 - invalid / (rows * len(checked))) * 100.0 if checked else 100.0

    # Cohérence inter-colonnes
    applicable = [
        r for r in mapping_rules
        if r.column_a in normalized.columns and r.column_b in normalized.columns
    ]
    inconsistent = sum(count_mapping_violations(normalized, r) for r in applicable)
    consistency = (
        (1 - inconsistent / (rows * len(applicable))) * 100.0 if applicable else 100.0
    )

    dims = [completeness, uniqueness, validity, consistency]
    return {
        "completeness": round(completeness, 3),
        "uniqueness": round(uniqueness, 3),
        "validity": round(validity, 3),
        "consistency": round(consistency, 3),
        "overall": round(sum(dims) / len(dims), 3),
        "rows": rows,
        "null_cells": null_cells,
        "text_numeric_columns": text_numeric,
    }


def validate_clean_table(
    table: str,
    frame: pd.DataFrame,
    *,
    key: list[str],
    structural: dict[str, dict[str, Any]],
    expected_numeric: list[str],
    range_rules: list[Any],
    mapping_rules: list[Any],
    product_rules: list[Any],
    rows_in: int,
    rows_removed: int,
) -> list[dict[str, Any]]:
    """Contrôles exécutés sur la table nettoyée (avant renommage des colonnes)."""
    checks: list[dict[str, Any]] = []

    def add(check: str, passed: bool, *, column: str | None = None, severity: str = "info",
            violations: int = 0, detail: str = "") -> None:
        checks.append({
            "id": f"{table}.{check}" + (f".{column}" if column else ""),
            "table": table, "check": check, "column": column, "passed": passed,
            "severity": "info" if passed else severity,
            "violations": violations, "detail": detail,
        })

    # Réconciliation des lignes : aucune ligne ne doit disparaître sans action tracée.
    add(
        "row_reconciliation", len(frame) == rows_in - rows_removed,
        severity="high", violations=abs(rows_in - rows_removed - len(frame)),
        detail=f"{rows_in} lignes en entrée − {rows_removed} supprimées tracées = "
               f"{rows_in - rows_removed} attendues ; {len(frame)} en sortie.",
    )

    dup = int(frame.duplicated().sum())
    add("no_exact_duplicates", dup == 0, severity="medium", violations=dup,
        detail=f"{dup} doublon(s) exact(s) restant(s).")

    if key and all(c in frame.columns for c in key):
        key_dup = int(frame.duplicated(subset=key).sum())
        key_null = int(frame[key].isna().any(axis=1).sum())
        add("business_key_unique", key_dup == 0, severity="high", violations=key_dup,
            detail=f"{key_dup} doublon(s) sur la clé {key}.")
        add("business_key_not_null", key_null == 0, severity="high", violations=key_null,
            detail=f"{key_null} ligne(s) avec clé nulle.")

    for column in expected_numeric:
        if column in frame.columns:
            ok = is_numeric(frame[column])
            add("numeric_dtype", ok, column=column, severity="high",
                detail=f"dtype = {frame[column].dtype}.")

    remaining_nulls = {
        str(c): int(frame[c].isna().sum()) for c in frame.columns
        if frame[c].isna().any() and str(c) not in structural
        and not pd.api.types.is_datetime64_any_dtype(frame[c])
    }
    add("no_unexpected_nulls", not remaining_nulls,
        severity="medium", violations=sum(remaining_nulls.values()),
        detail=f"Nulls restants hors cas structurels : {remaining_nulls or 'aucun'}.")

    for rule in range_rules:
        if rule.column in frame.columns:
            n = count_range_violations(frame, rule)
            add("range", n == 0, column=rule.column, severity="high", violations=n,
                detail=f"{n} valeur(s) hors de [{rule.min}, {rule.max}]."
                       + (" (règle automatique déduite du nom de la colonne)" if rule.auto else ""))

    for rule in mapping_rules:
        if rule.column_a in frame.columns and rule.column_b in frame.columns:
            n = count_mapping_violations(frame, rule)
            add("mapping_consistency", n == 0, column=f"{rule.column_a}->{rule.column_b}",
                severity="high", violations=n,
                detail=f"{n} ligne(s) où {rule.column_b} contredit {rule.column_a}.")

    for rule in product_rules:
        if {rule.column, rule.factor_a, rule.factor_b} <= set(frame.columns):
            n = count_product_violations(frame, rule)
            add("product_plausibility", n == 0, column=rule.column, severity="low",
                violations=n,
                detail=f"{n} ligne(s) où {rule.column} s'écarte de plus de "
                       f"{rule.tolerance_pct:g}% de {rule.factor_a} × {rule.factor_b}.")

    constant = [str(c) for c in frame.columns if frame[c].nunique(dropna=True) <= 1]
    add("no_constant_columns", not constant, severity="low", violations=len(constant),
        detail=f"Colonnes constantes (peu utiles pour l'analyse) : {constant or 'aucune'}.")

    return checks

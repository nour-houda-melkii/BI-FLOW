"""Contrôles de qualité : valeurs manquantes, doublons et incohérences simples."""

from __future__ import annotations

from typing import Any

import pandas as pd


def missing_by_column(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    total = len(frame)
    result = {}
    for column in frame.columns:
        missing = int(frame[column].isna().sum())
        result[str(column)] = {
            "count": missing,
            "pct": (missing / total * 100.0) if total else 0.0,
        }
    return result


def duplicate_profile(
    frame: pd.DataFrame,
    business_key: list[str] | None = None,
) -> dict[str, Any]:
    exact_count = int(frame.duplicated(keep="first").sum())
    total = len(frame)

    key = business_key or []
    unknown = sorted(set(key) - set(frame.columns))
    if unknown:
        raise ValueError(f"Colonnes de clé métier absentes : {unknown}")

    key_count = int(frame.duplicated(subset=key, keep="first").sum()) if key else 0

    return {
        "exact_count": exact_count,
        "exact_pct": (exact_count / total * 100.0) if total else 0.0,
        "business_key": key,
        "business_key_duplicate_count": key_count,
        "business_key_duplicate_pct": (key_count / total * 100.0) if total else 0.0,
    }


def simple_inconsistencies(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Signale des valeurs négatives dans les colonnes à nom évocateur.

    C'est un indicateur heuristique, pas une règle métier universelle.
    """
    issues: list[dict[str, Any]] = []
    keywords = ("amount", "charge", "fee", "revenue", "tenure", "quantity", "minutes")

    for column in frame.columns:
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        if not any(word in str(column).lower() for word in keywords):
            continue

        negative = series.lt(0).fillna(False)
        count = int(negative.sum())
        if count:
            issues.append({
                "column": str(column),
                "rule": "non_negative_by_name_heuristic",
                "affected_rows": count,
                "evidence": f"{count} valeur(s) négative(s) sur {len(frame)} ligne(s).",
            })
    return issues

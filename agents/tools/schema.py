"""Inférence de types, cardinalité et clés candidates."""

from __future__ import annotations

import pandas as pd


def infer_column_type(series: pd.Series) -> str:
    """Inférence déterministe, séparée du type pandas brut."""
    values = series.dropna()
    if values.empty:
        return "unknown"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # Tentative de conversion de dates pour les colonnes texte.
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        parsed_dates = pd.to_datetime(values, errors="coerce", format="mixed")
        if parsed_dates.notna().mean() >= 0.9:
            return "datetime"

        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().mean() >= 0.9:
            return "numeric"

        if values.nunique(dropna=True) <= min(20, max(2, int(len(values) * 0.05))):
            return "categorical"
        return "string"

    return str(series.dtype)


def column_cardinality(series: pd.Series) -> int:
    return int(series.nunique(dropna=True))


def candidate_keys(frame: pd.DataFrame) -> list[str]:
    """Colonnes non vides et uniques sur les lignes observées."""
    candidates: list[str] = []
    if frame.empty:
        return candidates

    for column in frame.columns:
        series = frame[column]
        if series.notna().all() and series.is_unique:
            candidates.append(str(column))
    return candidates


def declared_type(series: pd.Series) -> str:
    """Type déclaré par pandas lors de la lecture."""
    return str(series.dtype)

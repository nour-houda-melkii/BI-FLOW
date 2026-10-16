"""Statistiques descriptives déterministes."""

from __future__ import annotations

from typing import Any

import pandas as pd


def profile_column(
    series: pd.Series,
    *,
    rare_max_count: int = 5,
    rare_max_fraction: float = 0.01,
) -> dict[str, Any]:
    count = len(series)
    missing_count = int(series.isna().sum())
    result: dict[str, Any] = {
        "count": count,
        "missing_count": missing_count,
        "missing_pct": (missing_count / count * 100.0) if count else 0.0,
        "cardinality": int(series.nunique(dropna=True)),
    }

    non_null = series.dropna()
    if non_null.empty:
        return result

    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        if not numeric.empty:
            result["numeric"] = {
                "min": float(numeric.min()),
                "max": float(numeric.max()),
                "mean": float(numeric.mean()),
                "median": float(numeric.median()),
                "std": float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0,
                "q1": float(numeric.quantile(0.25)),
                "q3": float(numeric.quantile(0.75)),
            }

    elif pd.api.types.is_datetime64_any_dtype(series):
        dates = pd.to_datetime(non_null, errors="coerce").dropna()
        if not dates.empty:
            result["datetime"] = {
                "min": dates.min().isoformat(),
                "max": dates.max().isoformat(),
            }

    else:
        counts = non_null.astype("string").value_counts(dropna=True)
        result["categorical"] = {
            "top_values": [
                {"value": str(value), "count": int(n)}
                for value, n in counts.head(10).items()
            ]
        }
        threshold = max(rare_max_count, int(len(non_null) * rare_max_fraction))
        result["rare_values"] = [
            {"value": str(value), "count": int(n)}
            for value, n in counts.items()
            if n <= threshold
        ][:50]

    return result

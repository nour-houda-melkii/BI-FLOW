"""Calcul déterministe des KPI (pandas uniquement, aucun LLM).

Le LLM ne calcule JAMAIS un chiffre : il n'intervient qu'en aval, pour rédiger la synthèse.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any, Sequence

import pandas as pd

if TYPE_CHECKING:  # import de typage seulement : évite un import circulaire avec agents.kpi
    from agents.kpi.catalog import KPIDefinition

_POSITIVE = {"yes", "y", "true", "1", "churned", "oui"}
_NEGATIVE = {"no", "n", "false", "0", "retained", "non"}


def norm(name: Any) -> str:
    """'Monthly_Charges' / 'monthly charges' / 'MonthlyCharges' -> 'monthlycharges'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def as_flag(series: pd.Series) -> pd.Series | None:
    """Convertit une colonne binaire (0/1, bool, Yes/No...) en 0.0/1.0 ; None si ce n'en est pas une."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        values = set(series.dropna().unique())
        return series.astype(float) if values and values <= {0, 1} else None
    text = series.astype("string").str.strip().str.lower()
    if text.notna().sum() == 0 or not text.dropna().isin(_POSITIVE | _NEGATIVE).all():
        return None
    return text.isin(_POSITIVE).astype(float).where(text.notna())


def _role_ok(role: str, series: pd.Series) -> bool:
    if role == "flag":
        return as_flag(series) is not None
    if role == "id":
        return True
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def resolve_inputs(defn: KPIDefinition, df: pd.DataFrame) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Associe chaque rôle du KPI à une colonne réelle. Retourne (colonnes trouvées, rôles manquants)."""
    index = {norm(c): c for c in df.columns}
    found: dict[str, str] = {}
    missing: dict[str, list[str]] = {}
    for role, aliases in defn.inputs.items():
        column = next(
            (index[norm(a)] for a in aliases
             if norm(a) in index and _role_ok(role, df[index[norm(a)]])),
            None,
        )
        if column is None:
            missing[role] = aliases
        else:
            found[role] = column
    return found, missing


def compute_value(defn: KPIDefinition, df: pd.DataFrame, cols: dict[str, str]) -> float | None:
    """Valeur du KPI sur `df` (arrondie à 2 décimales) ; None si indéfinie."""
    agg = defn.agg
    if agg == "count_distinct":
        result: Any = df[cols["id"]].nunique(dropna=True)
    elif agg == "mean":
        result = df[cols["value"]].mean()
    elif agg == "sum":
        result = df[cols["value"]].sum()
    elif agg in {"rate", "count_flag", "sum_where"}:
        flag = as_flag(df[cols["flag"]])
        if flag is None:  # colonne vide ou non binaire sur ce sous-ensemble
            return None
        if agg == "rate":
            result = flag.mean() * 100
        elif agg == "count_flag":
            result = flag.sum()
        else:
            result = df.loc[flag == 1, cols["value"]].sum()
    elif agg == "ratio":
        denominator = df[cols["den"]].sum()
        result = df[cols["num"]].sum() / denominator * defn.scale if denominator else None
    else:
        return None

    if result is None or pd.isna(result) or not math.isfinite(float(result)):
        return None
    return round(float(result), 2)


def pick_dimensions(table_meta: dict[str, Any], df: pd.DataFrame, max_segments: int,
                    preferred: Sequence[str] = ()) -> list[str]:
    """Colonnes de segmentation : dimensions non personnelles, à faible cardinalité."""
    names = []
    for col in table_meta.get("columns", []):
        if col.get("semantic_role") != "dimension" or col.get("is_pii") or col.get("pii_treatment"):
            continue
        name = col.get("name")
        if name in df.columns and 2 <= df[name].nunique(dropna=True) <= max_segments:
            names.append(name)
    preferred = list(preferred)
    return sorted(names, key=lambda n: preferred.index(n) if n in preferred else len(preferred))


def compute_breakdown(defn: KPIDefinition, df: pd.DataFrame, cols: dict[str, str],
                      dimensions: list[str], min_rows: int) -> dict[str, list[dict[str, Any]]]:
    """KPI ventilé par segment. Les segments de moins de `min_rows` lignes sont ignorés."""
    used = set(cols.values())
    out: dict[str, list[dict[str, Any]]] = {}
    for dim in dimensions:
        if dim in used:
            continue
        # ventiler un KPI de churn par une colonne de churn (ex. churn_label) est tautologique
        if defn.category == "Churn" and norm(dim).startswith("churn"):
            continue
        rows = []
        for segment, part in df.groupby(dim, dropna=True, observed=True):
            if len(part) >= min_rows:
                rows.append({"segment": str(segment),
                             "value": compute_value(defn, part, cols), "rows": len(part)})
        if rows:
            out[dim] = rows
    return out
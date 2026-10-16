"""Détection d'anomalies univariées et multivariées."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def iqr_outliers(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    q1, q3 = numeric.quantile([0.25, 0.75])
    spread = q3 - q1
    if pd.isna(spread) or spread == 0:
        return pd.Series(False, index=series.index)
    return (numeric < q1 - multiplier * spread) | (numeric > q3 + multiplier * spread)


def zscore_outliers(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(False, index=series.index)
    zscores = (numeric - numeric.mean()) / std
    return zscores.abs() > threshold


def isolation_forest_outliers(
    frame: pd.DataFrame,
    *,
    contamination: float = 0.01,
    random_seed: int = 42,
) -> pd.Series:
    """Détecte les anomalies multivariées sur colonnes numériques complétées par médiane.

    Le remplissage est local à la matrice de calcul et ne modifie pas `frame`.
    """
    numeric = frame.select_dtypes(include="number")
    if len(numeric.columns) < 2 or len(frame) < 10:
        return pd.Series(False, index=frame.index)

    matrix = numeric.replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median()).fillna(0)

    model = IsolationForest(
        contamination=contamination,
        random_state=random_seed,
        n_jobs=1,
    )
    labels = model.fit_predict(matrix)
    return pd.Series(labels == -1, index=frame.index)

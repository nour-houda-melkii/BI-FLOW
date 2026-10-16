"""Schémas d'entrée et de sortie de l'ETL / Data Quality Agent.

Contrat avec les autres agents :
  - entrée  : `ETLRequest` = rapport du Data Profiler + configuration des sources ;
  - sortie  : `ETLReport`  = tables nettoyées (chemins), journal d'actions
              explicable (XAI), contrôles de validation et scores qualité.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# On réutilise les modèles du Profiler pour garder un contrat unique.
from agents.profiler.models import Issue, ProfilingReport, SourceConfig, TraceEntry


# --------------------------------------------------------------------------- #
# Règles métier configurables
# --------------------------------------------------------------------------- #
class RangeRule(BaseModel):
    """La colonne doit rester dans [min, max]."""

    model_config = ConfigDict(extra="forbid")

    column: str
    min: float | None = None
    max: float | None = None
    fix: Literal["report", "set_null"] = "report"
    auto: bool = Field(default=False, description="Règle déduite automatiquement du nom de colonne.")


class MappingRule(BaseModel):
    """Si `column_a == a_value` alors `column_b` doit valoir `mapping[a_value]`.

    Exemple : Churn Label "Yes" -> Churn Value 1.
    """

    model_config = ConfigDict(extra="forbid")

    column_a: str
    column_b: str
    mapping: dict[str, Any]
    fix: Literal["report", "derive_b"] = "report"


class ConditionalFillRule(BaseModel):
    """Remplit `column` avec `value` quand elle est nulle et `when_column == when_equals`.

    Exemple : Total Charges nul avec Tenure Months == 0 -> 0 (client sans facture).
    """

    model_config = ConfigDict(extra="forbid")

    column: str
    value: Any
    when_column: str
    when_equals: Any
    rationale: str = "Règle métier configurée."


class ProductRule(BaseModel):
    """Contrôle `column ≈ factor_a * factor_b` (tolérance en %). Signalement uniquement."""

    model_config = ConfigDict(extra="forbid")

    column: str
    factor_a: str
    factor_b: str
    tolerance_pct: float = Field(default=25.0, gt=0.0)


class StarSchemaSpec(BaseModel):
    """Surcharge manuelle de l'étoile proposée automatiquement (noms de colonnes de SORTIE)."""

    model_config = ConfigDict(extra="forbid")

    fact_name: str | None = None
    grain_key: str | None = None
    measures: list[str] | None = None
    dimensions: dict[str, list[str]] | None = Field(
        default=None, description='ex. {"dim_contract": ["contract", "payment_method"]}')
    exclude: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Entrée
# --------------------------------------------------------------------------- #
class ETLOptions(BaseModel):
    """Options de nettoyage. Toutes les stratégies sont déterministes."""

    model_config = ConfigDict(extra="forbid")

    # Sorties
    output_dir: str = "output/etl"
    write_outputs: bool = True
    output_format: Literal["csv", "parquet"] = "csv"
    standardize_column_names: bool = True

    # Normalisation du texte
    strip_whitespace: bool = True
    null_tokens: list[str] = Field(
        default_factory=lambda: ["", "n/a", "na", "null", "nan", "#n/a"],
        description="Chaînes (insensibles à la casse) traitées comme valeur manquante.",
    )
    harmonize_case: bool = True

    # Typage
    numeric_coercion_min_ratio: float = Field(default=0.9, gt=0.0, le=1.0)

    # Doublons
    drop_exact_duplicates: bool = True
    business_keys: dict[str, list[str]] = Field(
        default_factory=dict,
        description='Clé métier par table ; "*" = clé par défaut si les colonnes existent.',
    )
    key_duplicate_strategy: Literal["keep_first", "keep_last", "flag"] = "flag"

    # Valeurs manquantes
    numeric_missing_strategy: Literal["median", "mean", "zero", "keep"] = "median"
    categorical_missing_strategy: Literal["unknown", "mode", "keep"] = "unknown"
    unknown_label: str = "Unknown"
    not_applicable_label: str = "Not applicable"
    drop_column_missing_pct: float = Field(default=80.0, ge=0.0, le=100.0)
    drop_constant_columns: bool = False

    # Valeurs aberrantes : on signale par défaut, on ne supprime jamais.
    outlier_strategy: Literal["report", "clip"] = "report"
    iqr_multiplier: float = Field(default=1.5, gt=0.0)

    # PII
    pii_policy: Literal["keep", "hash", "drop"] = "hash"
    identifier_policy: Literal["keep", "hash"] = "keep"
    pii_salt_env: str = "BIFLOW_PII_SALT"

    # Règles métier
    preset: Literal["telco"] | None = None
    range_rules: list[RangeRule] = Field(default_factory=list)
    mapping_rules: list[MappingRule] = Field(default_factory=list)
    conditional_fills: list[ConditionalFillRule] = Field(default_factory=list)
    product_rules: list[ProductRule] = Field(default_factory=list)

    # Modèle en étoile
    build_star_schema: bool = True
    star_schemas: dict[str, StarSchemaSpec] = Field(
        default_factory=dict, description="Surcharge par table ; sinon proposition automatique.")

    # Résumé
    use_llm: bool = True
    min_quality_score: float = Field(default=90.0, ge=0.0, le=100.0)


class ETLRequest(BaseModel):
    """Demande transmise par l'Orchestrator."""

    model_config = ConfigDict(extra="forbid")

    profiling_report: ProfilingReport
    sources: list[SourceConfig] = Field(min_length=1)
    options: ETLOptions = Field(default_factory=ETLOptions)


# --------------------------------------------------------------------------- #
# Sortie
# --------------------------------------------------------------------------- #
class CleaningAction(BaseModel):
    """Une transformation appliquée. C'est l'unité d'explicabilité (XAI)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    step: int = Field(ge=0, description="Ordre d'exécution dans la table.")
    table: str
    column: str | None = None
    rule: str
    rows_affected: int = Field(ge=0)
    before: str = Field(description="État constaté avant l'action.")
    after: str = Field(description="État obtenu après l'action.")
    rationale: str = Field(description="Pourquoi cette correction a été appliquée.")
    source_issue_id: str | None = Field(
        default=None,
        description="Id du problème du Profiler résolu ; None si découvert par l'ETL.",
    )


class ValidationCheck(BaseModel):
    """Contrôle post-nettoyage."""

    model_config = ConfigDict(extra="forbid")

    id: str
    table: str
    check: str
    column: str | None = None
    passed: bool
    severity: Literal["info", "low", "medium", "high"] = "info"
    violations: int = Field(default=0, ge=0)
    detail: str = ""


class QualityScores(BaseModel):
    """Scores qualité 0-100, calculés sans LLM."""

    model_config = ConfigDict(extra="forbid")

    completeness: float = Field(ge=0.0, le=100.0)
    uniqueness: float = Field(ge=0.0, le=100.0)
    validity: float = Field(ge=0.0, le=100.0)
    consistency: float = Field(ge=0.0, le=100.0)
    overall: float = Field(ge=0.0, le=100.0)
    rows: int = Field(ge=0)
    null_cells: int = Field(ge=0)
    text_numeric_columns: list[str] = Field(
        default_factory=list,
        description="Colonnes numériques stockées comme texte.",
    )


class ColumnSummary(BaseModel):
    """Schéma de sortie d'une colonne, à destination du KPI / Analyst Agent."""

    model_config = ConfigDict(extra="forbid")

    original_name: str
    name: str
    dtype: str
    semantic_role: Literal["dimension", "measure", "id", "date", "unknown"] = "unknown"
    role_corrected_from: str | None = None
    is_pii: bool = False
    pii_treatment: str | None = None


class CleanedTable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    rows_in: int = Field(ge=0)
    rows_out: int = Field(ge=0)
    columns_in: int = Field(ge=0)
    columns_out: int = Field(ge=0)
    output_path: str | None = None
    quality_before: QualityScores
    quality_after: QualityScores
    columns: list[ColumnSummary] = Field(default_factory=list)

    @model_validator(mode="after")
    def _rows_consistent(self) -> CleanedTable:
        if self.rows_out > self.rows_in:
            raise ValueError("rows_out ne peut pas dépasser rows_in.")
        return self


class IssueResolution(BaseModel):
    """Lien explicite entre un problème du Profiler et ce que l'ETL en a fait."""

    model_config = ConfigDict(extra="forbid")

    issue_id: str
    status: Literal["resolved", "reported_only", "not_addressed"]
    action_ids: list[str] = Field(default_factory=list)
    note: str = ""


class StarDimension(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    key: str
    role: str
    columns: list[str]
    rows: int = Field(ge=0)
    rationale: str


class StarMeasure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    kind: Literal["metric", "flag"]
    suggested_aggregations: list[str]
    note: str = ""


class StarSchema(BaseModel):
    """Description du modèle en étoile : contrat pour le KPI / Analyst Agent."""

    model_config = ConfigDict(extra="forbid")

    table: str
    source: Literal["auto", "configured"]
    fact_name: str
    grain: str
    grain_reason: str
    fact_rows: int = Field(ge=0)
    measures: list[StarMeasure]
    dimensions: list[StarDimension]
    date_columns: list[str] = Field(default_factory=list)
    excluded_columns: list[dict[str, str]] = Field(default_factory=list)
    files: dict[str, str] = Field(default_factory=dict)
    check_ids: list[str] = Field(default_factory=list)
    mermaid: str = ""


class ETLReport(BaseModel):
    """Rapport remis à l'Orchestrator et à l'Auditor / XAI Agent."""

    model_config = ConfigDict(extra="forbid")

    business_goal: str
    status: Literal["success", "warning", "error"] = "success"
    tables: list[CleanedTable] = Field(default_factory=list)
    star_schemas: list[StarSchema] = Field(default_factory=list)
    actions: list[CleaningAction] = Field(default_factory=list)
    validation_checks: list[ValidationCheck] = Field(default_factory=list)
    issue_resolutions: list[IssueResolution] = Field(default_factory=list)
    remaining_issues: list[Issue] = Field(default_factory=list)
    llm_summary: str = ""
    trace: list[TraceEntry] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)

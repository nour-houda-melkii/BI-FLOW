"""Schémas d'entrée et de sortie du Data Profiler Agent."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SourceType(StrEnum):
    CSV = "csv"
    EXCEL = "excel"
    SQL = "sql"
    API = "api"


class SourceConfig(BaseModel):
    """Configuration non secrète d'une source de données.

    Les secrets ne doivent pas être inscrits directement dans `options`.
    Utiliser des variables d'environnement ou un gestionnaire de secrets.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Nom logique de la source")
    type: SourceType
    uri: str = Field(min_length=1, description="Chemin, URL ou chaîne de connexion")
    table: str | None = Field(
        default=None,
        description="Table SQL à lire, le cas échéant",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Options propres au connecteur : séparateur CSV, feuille Excel, etc.",
    )

    @field_validator("name", "uri")
    @classmethod
    def strip_required_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("La valeur ne peut pas être vide.")
        return value


class ProfileOptions(BaseModel):
    """Options de profilage déterministe."""

    model_config = ConfigDict(extra="forbid")

    sample_threshold_rows: int = Field(default=1_000_000, ge=1)
    sample_size: int = Field(default=100_000, ge=1)
    random_seed: int = 42

    rare_category_max_count: int = Field(default=5, ge=1)
    rare_category_max_fraction: float = Field(default=0.01, ge=0.0, le=1.0)

    anomaly_contamination: float = Field(default=0.01, gt=0.0, le=0.5)
    zscore_threshold: float = Field(default=3.0, gt=0.0)
    iqr_multiplier: float = Field(default=1.5, gt=0.0)

    # À fournir par l'appelant quand une clé métier est connue.
    business_keys: dict[str, list[str]] = Field(default_factory=dict)

    # Désactivable pour faciliter les tests ou limiter le coût de calcul.
    detect_anomalies: bool = True
    detect_relationships: bool = True
    detect_pii: bool = True


class ProfileRequest(BaseModel):
    """Demande de profilage transmise par l'Orchestrator."""

    model_config = ConfigDict(extra="forbid")

    business_goal: str = Field(min_length=1)
    sources: list[SourceConfig] = Field(min_length=1)
    options: ProfileOptions = Field(default_factory=ProfileOptions)

    @field_validator("business_goal")
    @classmethod
    def strip_business_goal(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("L'objectif métier ne peut pas être vide.")
        return value


class ColumnProfile(BaseModel):
    """Profil déterministe d'une colonne.

    `stats` contient des mesures calculées par les outils, jamais par le LLM.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    declared_type: str | None = None
    inferred_type: str
    semantic_role: Literal["dimension", "measure", "id", "date", "unknown"] = "unknown"

    missing_count: int = Field(default=0, ge=0)
    missing_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    cardinality: int = Field(default=0, ge=0)

    stats: dict[str, Any] = Field(default_factory=dict)
    rare_values: list[dict[str, Any]] = Field(default_factory=list)

    is_pii: bool = False
    pii_categories: list[str] = Field(default_factory=list)

    anomaly_count: int = Field(default=0, ge=0)
    # Les valeurs des colonnes PII doivent être masquées avant d'être ajoutées.
    anomaly_examples: list[Any] = Field(default_factory=list)


class DuplicateProfile(BaseModel):
    """Doublons détectés dans une table."""

    model_config = ConfigDict(extra="forbid")

    exact_count: int = Field(default=0, ge=0)
    exact_pct: float = Field(default=0.0, ge=0.0, le=100.0)

    business_key: list[str] = Field(default_factory=list)
    business_key_duplicate_count: int = Field(default=0, ge=0)
    business_key_duplicate_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class TableProfile(BaseModel):
    """Résumé de profilage d'une table."""

    model_config = ConfigDict(extra="forbid")

    name: str
    source_name: str
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    memory_bytes: int = Field(default=0, ge=0)

    sampled: bool = False
    sample_rows: int | None = Field(default=None, ge=0)
    columns: list[ColumnProfile] = Field(default_factory=list)
    duplicates: DuplicateProfile = Field(default_factory=DuplicateProfile)

    @model_validator(mode="after")
    def validate_column_count(self) -> TableProfile:
        if self.columns and len(self.columns) != self.column_count:
            raise ValueError(
                "column_count doit correspondre au nombre de profils de colonnes."
            )
        if self.sampled and self.sample_rows is None:
            raise ValueError("sample_rows est requis lorsque sampled=True.")
        return self


class RelationshipProfile(BaseModel):
    """Relation candidate entre deux colonnes."""

    model_config = ConfigDict(extra="forbid")

    table_a: str
    column_a: str
    table_b: str
    column_b: str
    relationship_type: Literal["1-1", "1-N", "N-1", "N-N", "unknown"] = "unknown"

    match_rate_pct: float = Field(ge=0.0, le=100.0)
    orphan_count_a: int = Field(default=0, ge=0)
    orphan_count_b: int = Field(default=0, ge=0)

    evidence: str = Field(
        description="Preuve chiffrée ou règle déterministe ayant produit la relation"
    )


class Issue(BaseModel):
    """Problème explicable et exploitable par les agents ETL et Auditor."""

    model_config = ConfigDict(extra="forbid")

    id: str
    severity: Literal["low", "medium", "high"]
    table: str
    column: str | None = None

    rule: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    affected_rows: int = Field(ge=0)
    suggested_action: str = Field(min_length=1)


class TraceEntry(BaseModel):
    """Événement de traçabilité d'un outil."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=datetime.now)
    tool: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = Field(ge=0.0)


class ProfilingReport(BaseModel):
    """Rapport remis à l'Orchestrator."""

    model_config = ConfigDict(extra="forbid")

    business_goal: str
    tables: list[TableProfile] = Field(default_factory=list)
    relationships: list[RelationshipProfile] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)

    llm_summary: str = ""
    status: Literal["success", "warning", "error"] = "success"
    trace: list[TraceEntry] = Field(default_factory=list)

    errors: list[str] = Field(
        default_factory=list,
        description="Erreurs lisibles par l'Orchestrator, sans exception non gérée.",
    )

"""Schémas d'entrée et de sortie du KPI & Semantic Agent.

Contrat avec les autres agents :
  - entrée : `KPIRequest` = rapport de l'ETL Agent (etl_report.json) ;
  - sortie : `KPIReport`  = KPI calculés + définitions + justifications + réserves sur la
             qualité des données, prêts pour le BI Analyst Agent et l'Auditor / XAI Agent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.profiler.models import TraceEntry


class KPIOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = "output/kpi"
    write_outputs: bool = True
    categories: list[str] | None = Field(
        default=None, description="Catégories de KPI à calculer ; None = toutes.")
    max_breakdown_dimensions: int = Field(default=3, ge=0)
    max_segments: int = Field(default=10, ge=2, description="Cardinalité max d'une dimension.")
    min_segment_rows: int = Field(default=30, ge=1, description="Taille min d'un segment.")
    min_quality_score: float = Field(default=90.0, ge=0.0, le=100.0)
    use_llm: bool = True


class KPIRequest(BaseModel):
    """Demande transmise par l'Orchestrator.

    `etl_report` = `ETLReport.model_dump(mode="json")` (ou le contenu de etl_report.json).
    """

    model_config = ConfigDict(extra="forbid")

    etl_report: dict[str, Any]
    options: KPIOptions = Field(default_factory=KPIOptions)


class SegmentValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment: str
    value: float | None
    rows: int = Field(ge=0)


class KPIResult(BaseModel):
    """Un KPI : valeur + définition + traçabilité (unité d'explicabilité pour le XAI)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    category: str
    definition: str
    formula: str
    unit: str = ""
    status: Literal["computed", "suspect", "not_computable"]
    value: float | None = None
    table: str | None = None
    rows_used: int = Field(default=0, ge=0)
    columns_used: dict[str, str] = Field(default_factory=dict)
    missing_inputs: dict[str, list[str]] = Field(
        default_factory=dict, description="Rôle -> alias de colonnes cherchés et introuvables.")
    breakdowns: dict[str, list[SegmentValue]] = Field(default_factory=dict)
    justification: str
    warnings: list[str] = Field(default_factory=list)


class KPIReport(BaseModel):
    """Rapport remis au BI Analyst Agent, à l'Orchestrator et à l'Auditor / XAI Agent."""

    model_config = ConfigDict(extra="forbid")

    business_goal: str = ""
    status: Literal["success", "warning", "error"] = "success"
    etl_status: str = "unknown"
    kpis: list[KPIResult] = Field(default_factory=list)
    caveats: list[str] = Field(
        default_factory=list, description="Réserves sur la qualité des données à garder en tête.")
    output_files: dict[str, str] = Field(default_factory=dict)
    llm_summary: str = ""
    trace: list[TraceEntry] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)
"""Teste le pipeline Profiler -> ETL (+ étoile) -> KPI sur N'IMPORTE QUEL fichier CSV.

Usage :
    python run_pipeline.py data/telco.csv --preset telco
    python run_pipeline.py chemin/vers/mon_fichier.csv
    python run_pipeline.py mon_fichier.csv --key subscriber_id --sep ";"
    python run_pipeline.py data/telco.csv --preset telco --no-kpi
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from agents.etl import ETLOptions, ETLRequest, run_with_data
from agents.kpi import KPIOptions, KPIRequest
from agents.kpi import run as compute_kpis
from agents.profiler.agent import run as profile
from agents.profiler.models import ProfileOptions, ProfileRequest, SourceConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Profiler -> ETL -> étoile -> KPI sur un CSV")
    parser.add_argument("csv")
    parser.add_argument("--name", help="nom logique de la table (défaut : nom du fichier)")
    parser.add_argument("--goal", default="Analyse BI telecom")
    parser.add_argument("--preset", choices=["telco"], default=None)
    parser.add_argument("--key", help="colonne clé métier (ex. customer_id)")
    parser.add_argument("--sep", default=",")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--out", default="output")
    parser.add_argument("--no-kpi", action="store_true", help="s'arrêter après l'ETL")
    args = parser.parse_args()

    path = Path(args.csv)
    name = args.name or re.sub(r"\W+", "_", path.stem).strip("_").lower()
    source = SourceConfig(name=name, type="csv", uri=str(path),
                          options={"sep": args.sep, "encoding": args.encoding})
    keys = {name: [args.key]} if args.key else {}

    print(f"1) PROFILER sur {path} ...")
    profiling = profile(ProfileRequest(
        business_goal=args.goal, sources=[source],
        options=ProfileOptions(business_keys=keys)))
    print(f"   statut : {profiling.status} | {len(profiling.issues)} problème(s) signalé(s)")
    for issue in profiling.issues:
        print(f"   - [{issue.severity}] {issue.table}.{issue.column or ''} : {issue.rule} "
              f"({issue.affected_rows} ligne(s))")

    out_dir = Path(args.out) / name
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "profiler_output.json").write_text(
        profiling.model_dump_json(indent=2), encoding="utf-8")

    print("\n2) ETL ...")
    report, frames = run_with_data(ETLRequest(
        profiling_report=profiling, sources=[source],
        options=ETLOptions(preset=args.preset, business_keys=keys, use_llm=False,
                           output_dir=str(out_dir))))
    print(f"   statut : {report.status}")
    for table in report.tables:
        print(f"   {table.name} : {table.rows_in} -> {table.rows_out} lignes | qualité "
              f"{table.quality_before.overall} -> {table.quality_after.overall}")
    print(f"   {len(report.actions)} action(s) :")
    for action in report.actions:
        if action.rows_affected:
            print(f"   - {action.rule}"
                  f"{'.' + action.column if action.column else ''} : {action.rows_affected} ligne(s)")
    for error in report.errors:
        print(f"   ! {error}")

    for star in report.star_schemas:
        print(f"\n3) MODÈLE EN ÉTOILE : {star.fact_name}  ({star.grain})")
        print(f"   mesures    : {[m.column for m in star.measures]}")
        for dim in star.dimensions:
            print(f"   {dim.name:<16}{dim.rows:>6} lignes  {dim.columns}")
        if star.date_columns:
            print(f"   dates      : {star.date_columns} -> dim_date")
        print(f"   exclues    : {[e['column'] + ' (' + e['reason'] + ')' for e in star.excluded_columns]}")
        ok = sum(1 for c in report.validation_checks if c.check.startswith('star') and c.passed)
        total = sum(1 for c in report.validation_checks if c.check.startswith('star'))
        print(f"   contrôles étoile : {ok}/{total} OK")

    if report.remaining_issues:
        print("\n4) À VALIDER PAR UN HUMAIN :")
        for issue in report.remaining_issues:
            print(f"   - [{issue.severity}] {issue.rule} : {issue.evidence}")

    kpi_status = "skipped"
    if not args.no_kpi and report.status != "error":
        print("\n5) KPI ...")
        kpi_report = compute_kpis(
            KPIRequest(etl_report=report.model_dump(mode="json"),
                       options=KPIOptions(output_dir=str(out_dir / "kpi"), use_llm=False)),
            frames=frames)
        kpi_status = kpi_report.status
        print(f"   statut : {kpi_report.status}")
        for kpi in kpi_report.kpis:
            if kpi.status != "not_computable":
                flag = "  [SUSPECT]" if kpi.status == "suspect" else ""
                print(f"   - {kpi.name:<34}{kpi.value:>12} {kpi.unit}{flag}")
        skipped = [k.id for k in kpi_report.kpis if k.status == "not_computable"]
        if skipped:
            print(f"   non calculables avec ces données : {skipped}")
        for caveat in kpi_report.caveats:
            print(f"   ? {caveat}")
        for error in kpi_report.errors:
            print(f"   ! {error}")

    print(f"\nFichiers dans : {out_dir}")
    return 0 if report.status != "error" and kpi_status != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
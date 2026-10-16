"""CLI : python -m agents.etl --profile profiler_output.json --source telco=csv:data/telco.csv"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agents.etl.agent import run
from agents.etl.models import ETLOptions, ETLRequest
from agents.profiler.models import ProfilingReport, SourceConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ETL / Data Quality Agent")
    parser.add_argument("--profile", required=True, help="JSON produit par le Data Profiler")
    parser.add_argument("--source", action="append", required=True,
                        help="nom=type:chemin  (ex. telco=csv:data/telco.csv)")
    parser.add_argument("--preset", choices=["telco"], default=None)
    parser.add_argument("--output-dir", default="output/etl")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--no-llm", action="store_true", help="synthèse déterministe uniquement")
    args = parser.parse_args(argv)

    profile = ProfilingReport.model_validate_json(Path(args.profile).read_text(encoding="utf-8"))
    sources = []
    for spec in args.source:
        name, _, rest = spec.partition("=")
        kind, _, uri = rest.partition(":")
        sources.append(SourceConfig(name=name, type=kind, uri=uri))

    request = ETLRequest(
        profiling_report=profile, sources=sources,
        options=ETLOptions(preset=args.preset, output_dir=args.output_dir,
                           output_format=args.format, use_llm=not args.no_llm),
    )
    report = run(request)

    print(json.dumps({
        "status": report.status,
        "tables": [{"name": t.name, "rows": f"{t.rows_in}->{t.rows_out}",
                    "quality": f"{t.quality_before.overall}->{t.quality_after.overall}",
                    "output": t.output_path} for t in report.tables],
        "actions": len(report.actions), "errors": report.errors,
    }, indent=2, ensure_ascii=False))
    print("\n" + report.llm_summary)
    return 0 if report.status != "error" else 1


if __name__ == "__main__":
    sys.exit(main())

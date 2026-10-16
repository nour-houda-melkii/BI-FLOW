# ETL / Data Quality Agent (BIFlow)

Transforme les données brutes + le rapport du Data Profiler en tables propres,
avec un journal d'actions explicable pour le BI Auditor / XAI Agent.

## Fichiers ajoutés (aucune dépendance nouvelle)
```
agents/etl/           models.py  agent.py  presets.py  prompts.py  __main__.py  __init__.py
agents/tools/         cleaning.py  validation.py  modeling.py
tests/etl/            test_cleaning.py  test_validation.py  test_agent.py  test_modeling.py  fixtures/profiler_output.json
conftest.py           (vide, permet de lancer `pytest` depuis la racine)
run_pipeline.py       teste Profiler -> ETL -> étoile sur N'IMPORTE QUEL csv
```

## Lancer
```bash
python -m agents.etl --profile profiler_output.json --source telco=csv:data/telco.csv --preset telco --no-llm
# -> output/etl/telco_clean.csv  +  output/etl/etl_report.json
```

## Tester
```bash
python -m pytest tests/etl -v      # 94 tests ETL
python -m pytest tests -q          # tout le projet (profiler inclus)
```

## Brancher dans l'Orchestrator
```python
from agents.profiler import run as profile
from agents.profiler.models import ProfileRequest, SourceConfig
from agents.etl import run as etl, run_with_data, ETLRequest, ETLOptions

sources = [SourceConfig(name="telco", type="csv", uri="data/telco.csv")]
profiling = profile(ProfileRequest(business_goal="Analyser le churn", sources=sources))
report, frames = run_with_data(ETLRequest(
    profiling_report=profiling, sources=sources, options=ETLOptions(preset="telco")))
# frames["telco"]  -> DataFrame propre (colonnes snake_case) pour le KPI Agent
# report           -> ETLReport (actions, checks, scores, rôles sémantiques corrigés)
```

## Sorties
```
output/etl/
├── telco_clean.csv              table plate nettoyée (staging)
├── etl_report.json              rapport complet (actions, checks, scores, star_schemas)
└── model/telco/                 MODÈLE EN ÉTOILE
    ├── fact_customer.csv        clés étrangères + mesures
    ├── dim_customer.csv  dim_churn.csv  dim_contract.csv  dim_location.csv  dim_service.csv
    └── star_schema.mmd          diagramme (à coller sur mermaid.live)
```
L'étoile est proposée automatiquement (grain, mesures, dimensions) puis vérifiée :
lignes, clés nulles/orphelines, couverture des colonnes, sommes des mesures et
re-jointure sans perte. Pour la corriger : `ETLOptions(star_schemas={"table": StarSchemaSpec(...)})`.
Pour la désactiver : `ETLOptions(build_star_schema=False)`.

## Tester n'importe quel CSV
```bash
python run_pipeline.py data/telco.csv --preset telco
python run_pipeline.py mon_fichier.csv --key subscriber_id
```

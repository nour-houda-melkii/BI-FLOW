"""Presets de règles métier. Le preset `telco` correspond au jeu IBM Telco Churn."""

from __future__ import annotations

from agents.etl.models import (
    ConditionalFillRule,
    ETLOptions,
    MappingRule,
    ProductRule,
    RangeRule,
)

_INTERNET_DEPENDENT = [
    "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies",
]


def telco_rules() -> dict:
    return {
        "business_keys": {"*": ["CustomerID"]},
        "range_rules": [
            RangeRule(column="Tenure Months", min=0),
            RangeRule(column="Monthly Charges", min=0),
            RangeRule(column="Total Charges", min=0),
            RangeRule(column="Churn Score", min=0, max=100),
            RangeRule(column="CLTV", min=0),
            RangeRule(column="Churn Value", min=0, max=1),
        ],
        "mapping_rules": [
            MappingRule(column_a="Churn Label", column_b="Churn Value",
                        mapping={"Yes": 1, "No": 0}),
            MappingRule(column_a="Phone Service", column_b="Multiple Lines",
                        mapping={"No": "No phone service"}),
            *[
                MappingRule(column_a="Internet Service", column_b=col,
                            mapping={"No": "No internet service"})
                for col in _INTERNET_DEPENDENT
            ],
        ],
        "conditional_fills": [
            ConditionalFillRule(
                column="Total Charges", value=0.0,
                when_column="Tenure Months", when_equals=0,
                rationale="Un client à 0 mois d'ancienneté n'a encore reçu aucune facture : "
                          "le total cumulé vaut 0, pas la médiane des autres clients.",
            ),
        ],
        "product_rules": [
            ProductRule(column="Total Charges", factor_a="Tenure Months",
                        factor_b="Monthly Charges", tolerance_pct=25.0),
        ],
    }


def apply_preset(options: ETLOptions) -> ETLOptions:
    """Retourne des options enrichies par le preset ; les règles explicites priment."""
    if options.preset == "telco":
        rules = telco_rules()
        update = {}
        for field, preset_value in rules.items():
            current = getattr(options, field)
            if isinstance(current, dict):
                update[field] = {**preset_value, **current}
            elif not current:
                update[field] = preset_value
        return options.model_copy(update=update)
    return options

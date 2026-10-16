import pytest
from pydantic import ValidationError

from agents.profiler.models import (
    ColumnProfile,
    ProfileRequest,
    SourceConfig,
    SourceType,
    TableProfile,
)


def test_profile_request_accepts_csv_source() -> None:
    request = ProfileRequest(
        business_goal="Analyser le churn client",
        sources=[
            SourceConfig(
                name="telco",
                type=SourceType.CSV,
                uri="data/telco.csv",
            )
        ],
    )

    assert request.business_goal == "Analyser le churn client"
    assert request.options.sample_threshold_rows == 1_000_000


def test_profile_request_rejects_empty_sources() -> None:
    with pytest.raises(ValidationError):
        ProfileRequest(business_goal="Analyser le churn", sources=[])


def test_table_profile_checks_column_count() -> None:
    with pytest.raises(ValidationError):
        TableProfile(
            name="customers",
            source_name="telco",
            row_count=10,
            column_count=2,
            columns=[
                ColumnProfile(
                    name="customer_id",
                    inferred_type="string",
                    semantic_role="id",
                )
            ],
        )


def test_sampled_table_requires_sample_size() -> None:
    with pytest.raises(ValidationError):
        TableProfile(
            name="customers",
            source_name="telco",
            row_count=2_000_000,
            column_count=0,
            sampled=True,
        )

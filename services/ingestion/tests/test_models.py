from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from weathergpt_ingestion.models import NormalizedObservation, Provenance, SourceName


def test_observation_rejects_invalid_coordinates():
    with pytest.raises(ValidationError):
        NormalizedObservation(
            observed_at=datetime(2026, 9, 12, 12, 0, tzinfo=UTC),
            source=SourceName.OPEN_METEO,
            external_id="bad",
            location_name="bad",
            latitude=120,
            longitude=72,
            provenance=Provenance(source=SourceName.OPEN_METEO, connector="test"),
            raw_payload={},
        )


def test_observation_rejects_invalid_humidity():
    with pytest.raises(ValidationError):
        NormalizedObservation(
            observed_at=datetime(2026, 9, 12, 12, 0, tzinfo=UTC),
            source=SourceName.OPEN_METEO,
            external_id="bad",
            location_name="bad",
            latitude=19,
            longitude=72,
            humidity_pct=122,
            provenance=Provenance(source=SourceName.OPEN_METEO, connector="test"),
            raw_payload={},
        )

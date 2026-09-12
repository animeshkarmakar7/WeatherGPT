from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class SourceName(StrEnum):
    OPEN_METEO = "open_meteo"
    NOAA = "noaa"
    IMD = "imd"
    WIS2 = "wis2"


class IngestionStatus(StrEnum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEGRADED = "degraded"


class QualityFlag(StrEnum):
    LAST_KNOWN_GOOD = "last_known_good"
    PARTIAL_PAYLOAD = "partial_payload"
    UPSTREAM_DELAYED = "upstream_delayed"
    VALIDATION_WARNING = "validation_warning"


class Provenance(BaseModel):
    source: SourceName
    connector: str
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_url: str | None = None
    license: str | None = None
    attribution: str | None = None


class RawWeatherEvent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    source: SourceName
    topic: str
    external_id: str
    location_name: str
    latitude: float
    longitude: float
    observed_at: datetime
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any]
    provenance: Provenance
    quality_flags: list[QualityFlag] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_coordinates(self) -> "RawWeatherEvent":
        _validate_coordinates(self.latitude, self.longitude)
        return self

    @field_validator("location_name")
    @classmethod
    def normalize_location_name(cls, value: str) -> str:
        return value.strip().lower()


class NormalizedObservation(BaseModel):
    observed_at: datetime
    source: SourceName
    external_id: str
    location_name: str
    latitude: float
    longitude: float
    temp_c: float | None = None
    wind_speed_kph: float | None = None
    wind_direction_deg: float | None = None
    humidity_pct: float | None = None
    precipitation_mm: float | None = None
    pressure_hpa: float | None = None
    weather_code: str | None = None
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    provenance: Provenance
    raw_payload: dict[str, Any]

    @model_validator(mode="after")
    def validate_observation(self) -> "NormalizedObservation":
        _validate_coordinates(self.latitude, self.longitude)
        _validate_percent(self.humidity_pct, "humidity_pct")
        _validate_non_negative(self.wind_speed_kph, "wind_speed_kph")
        _validate_non_negative(self.precipitation_mm, "precipitation_mm")
        if self.wind_direction_deg is not None and not 0 <= self.wind_direction_deg <= 360:
            raise ValueError("wind_direction_deg must be between 0 and 360")
        return self

    @field_validator("location_name")
    @classmethod
    def normalize_location_name(cls, value: str) -> str:
        return value.strip().lower()


class DeadLetterEvent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    source: str
    topic: str
    error_type: str
    error_message: str
    payload: dict[str, Any]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class IngestionRun(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    source: SourceName
    connector: str
    status: IngestionStatus = IngestionStatus.STARTED
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    records_fetched: int = 0
    records_published: int = 0
    error_message: str | None = None


def _validate_coordinates(latitude: float, longitude: float) -> None:
    if not -90 <= latitude <= 90:
        raise ValueError("latitude must be between -90 and 90")
    if not -180 <= longitude <= 180:
        raise ValueError("longitude must be between -180 and 180")


def _validate_percent(value: float | None, field_name: str) -> None:
    if value is not None and not 0 <= value <= 100:
        raise ValueError(f"{field_name} must be between 0 and 100")


def _validate_non_negative(value: float | None, field_name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} must be non-negative")

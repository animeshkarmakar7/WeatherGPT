from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class IntentType(StrEnum):
    WEATHER_CURRENT = "weather_current"
    WEATHER_FORECAST = "weather_forecast"
    ALERT = "alert"
    CLIMATE = "climate"
    GENERAL = "general"


class WeatherAspect(StrEnum):
    RAIN = "rain"
    TEMPERATURE = "temperature"
    WIND = "wind"
    HUMIDITY = "humidity"
    GENERAL = "general"


class QueryClassification(BaseModel):
    intent: IntentType = IntentType.WEATHER_FORECAST
    location: str | None = None
    target_date: str | None = "current"
    aspect: WeatherAspect = WeatherAspect.GENERAL
    language: str = "en"
    confidence: float = 0.95


class WeatherDataFact(BaseModel):
    location: str
    latitude: float
    longitude: float
    query_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    target_date: str = "current"
    temp_c: float | None = None
    temp_min_c: float | None = None
    temp_max_c: float | None = None
    humidity_pct: float | None = None
    precipitation_mm: float | None = None
    precipitation_probability_pct: float | None = None
    wind_speed_kph: float | None = None
    weather_code: str | None = None
    condition_description: str = "Unknown"
    will_rain: bool = False
    source: str = "unavailable"
    source_url: str | None = None
    observed_at: datetime | None = None
    fetched_at: datetime | None = None
    freshness: str = "unknown"
    cached: bool = False


class StructuredWeatherResponse(BaseModel):
    location: str
    target_date: str
    summary: str
    will_rain: bool
    precipitation_probability_pct: float | None = None
    temp_c: float | None = None
    temp_min_c: float | None = None
    temp_max_c: float | None = None
    humidity_pct: float | None = None
    wind_speed_kph: float | None = None
    conditions: str
    confidence: float = 0.90
    data_sources: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    observed_at: datetime | None = None
    fetched_at: datetime | None = None
    freshness: str = "unknown"


class ChatMessageRequest(BaseModel):
    session_id: str | None = None
    message: str
    location_hint: str | None = None


class ChatMessageResponse(BaseModel):
    session_id: str
    message_id: UUID = Field(default_factory=uuid4)
    response_text: str
    structured_data: StructuredWeatherResponse | None = None
    intent: IntentType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

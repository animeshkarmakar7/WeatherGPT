from typing import Any

from .models import NormalizedObservation, RawWeatherEvent, SourceName


def normalize_event(event: RawWeatherEvent) -> NormalizedObservation:
    if event.source == SourceName.OPEN_METEO:
        current = event.payload.get("current") or {}
        return NormalizedObservation(
            observed_at=event.observed_at,
            source=event.source,
            external_id=event.external_id,
            location_name=event.location_name,
            latitude=event.latitude,
            longitude=event.longitude,
            temp_c=_number(current.get("temperature_2m")),
            wind_speed_kph=_number(current.get("wind_speed_10m")),
            wind_direction_deg=_number(current.get("wind_direction_10m")),
            humidity_pct=_number(current.get("relative_humidity_2m")),
            precipitation_mm=_number(current.get("precipitation")),
            pressure_hpa=_number(current.get("surface_pressure")),
            weather_code=_string(current.get("weather_code")),
            quality_flags=event.quality_flags,
            provenance=event.provenance,
            raw_payload=event.payload,
        )

    if event.source == SourceName.NOAA:
        periods = event.payload.get("properties", {}).get("periods") or []
        period = periods[0] if periods else {}
        temp_c = None
        if period.get("temperature") is not None:
            temp_c = (float(period["temperature"]) - 32.0) * 5.0 / 9.0
        return NormalizedObservation(
            observed_at=event.observed_at,
            source=event.source,
            external_id=event.external_id,
            location_name=event.location_name,
            latitude=event.latitude,
            longitude=event.longitude,
            temp_c=temp_c,
            wind_speed_kph=_parse_noaa_wind_kph(period.get("windSpeed")),
            wind_direction_deg=None,
            humidity_pct=_number(period.get("relativeHumidity", {}).get("value")),
            precipitation_mm=None,
            pressure_hpa=None,
            weather_code=_string(period.get("shortForecast")),
            quality_flags=event.quality_flags,
            provenance=event.provenance,
            raw_payload=event.payload,
        )

    raise ValueError(f"no normalizer registered for source {event.source}")


def _number(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _string(value: Any) -> str | None:
    return None if value is None else str(value)


def _parse_noaa_wind_kph(value: str | None) -> float | None:
    if not value:
        return None
    first_token = value.split()[0]
    try:
        mph = float(first_token)
    except ValueError:
        return None
    return mph * 1.60934

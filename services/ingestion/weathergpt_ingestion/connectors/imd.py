from datetime import UTC, datetime

from ..config import Settings
from ..models import Provenance, RawWeatherEvent, SourceName
from ..resilience import CircuitOpenError
from ..topics import IMD_CURRENT
from .base import WeatherConnector


class ImdCurrentWeatherConnector(WeatherConnector):
    @property
    def name(self) -> str:
        return "imd_current_weather"

    @property
    def topic(self) -> str:
        return IMD_CURRENT

    async def fetch_current(self, location_name: str, latitude: float, longitude: float) -> RawWeatherEvent:
        station_id = self.settings.imd_station_ids.get(location_name.lower())
        if not station_id:
            raise ValueError(
                f"no IMD station id configured for '{location_name}'; "
                "set WEATHERGPT_IMD_STATION_IDS as a JSON object"
            )

        key = self.cache_key(location_name, latitude, longitude)

        async def _fetch() -> RawWeatherEvent:
            url = f"{self.settings.imd_base_url}/api/v1/current_wx"
            response = await self.client.get(url, params={"id": station_id})
            response.raise_for_status()
            payload = response.json()
            record = _first_record(payload)
            observed_at = _parse_observed_at(record)
            source_latitude = _number(record.get("Latitude")) or latitude
            source_longitude = _number(record.get("Longitude")) or longitude
            external_id = f"imd:{station_id}:{observed_at.isoformat()}"
            event = RawWeatherEvent(
                source=SourceName.IMD,
                topic=self.topic,
                external_id=external_id,
                location_name=location_name,
                latitude=source_latitude,
                longitude=source_longitude,
                observed_at=observed_at,
                payload=payload,
                provenance=Provenance(
                    source=SourceName.IMD,
                    connector=self.name,
                    source_url=str(response.url),
                    attribution="India Meteorological Department",
                ),
            )
            await self.cache.set(key, event)
            return event

        try:
            return await self.breaker.call(_fetch)
        except CircuitOpenError:
            cached = await self.cache.get(key)
            if cached:
                return cached
            raise
        except Exception:
            cached = await self.cache.get(key)
            if cached:
                return cached
            raise


def _first_record(payload: object) -> dict:
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0]
        if isinstance(data, dict):
            return data
        return payload
    raise ValueError("unexpected IMD current weather response shape")


def _parse_observed_at(record: dict) -> datetime:
    date_value = record.get("Date of Observation") or record.get("Date")
    time_value = record.get("Time of Observation") or record.get("Time")
    if date_value and time_value:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S"):
            try:
                return datetime.strptime(f"{date_value} {time_value}", fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
    if date_value:
        for fmt in ("%Y-%m-%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(str(date_value), fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
    return datetime.now(UTC)


def _number(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

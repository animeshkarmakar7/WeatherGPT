import json
import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from redis.asyncio import Redis

from .config import ChatSettings
from .models import WeatherDataFact

logger = logging.getLogger(__name__)

_WMO_CONDITIONS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall", 77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers", 95: "Thunderstorm",
    96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


def _wmo_to_condition(code: str | int | None) -> str:
    try:
        return _WMO_CONDITIONS.get(int(code), "Unknown")
    except (ValueError, TypeError):
        return "Unknown"


def _freshness(observed_at: datetime | None, limit_seconds: int) -> str:
    if observed_at is None:
        return "unknown"
    value = observed_at.replace(tzinfo=UTC) if observed_at.tzinfo is None else observed_at
    age = max(0.0, (datetime.now(UTC) - value).total_seconds())
    return "fresh" if age <= limit_seconds else "stale"


class WeatherDataQueryService:
    def __init__(self, settings: ChatSettings, redis: Redis | None = None) -> None:
        self.settings = settings
        self.redis = redis
        db_url = settings.database_url.replace("postgresql+psycopg://", "postgresql://")
        self.pool = AsyncConnectionPool(
            conninfo=db_url,
            min_size=settings.database_pool_min_size,
            max_size=settings.database_pool_max_size,
            open=False,
        )

    async def start(self) -> None:
        await self.pool.open()

    async def close(self) -> None:
        await self.pool.close()

    async def ping(self) -> bool:
        try:
            async with self.pool.connection() as conn:
                cur = await conn.execute("SELECT 1")
                return await cur.fetchone() is not None
        except Exception:
            return False

    async def _resolve_coordinates(self, location: str) -> tuple[float, float, str]:
        norm = location.strip().lower()
        if norm in self.settings.default_cities:
            lat, lon = self.settings.default_cities[norm]
            return lat, lon, norm
        async with httpx.AsyncClient(timeout=4.0) as client:
            response = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": norm, "count": 5, "language": "en", "format": "json"},
            )
            response.raise_for_status()
            results = response.json().get("results") or []
            if not results:
                raise ValueError(f"No verified location found for '{location}'")
            item = results[0]
            return float(item["latitude"]), float(item["longitude"]), str(item.get("name", location))

    async def get_weather_data(self, location: str, target_date: str = "current") -> WeatherDataFact:
        norm_loc = location.strip().lower()
        lat, lon, resolved_name = await self._resolve_coordinates(norm_loc)
        if target_date not in {"current", "today", "now"} and self.redis:
            cache_key = f"weather:read:{norm_loc}:{target_date}"
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    data = json.loads(cached)
                    data["cached"] = True
                    return WeatherDataFact.model_validate(data)
            except Exception as exc:
                logger.warning("Forecast cache read failed: %s", exc)
        fact = await self._query_verified_data(resolved_name, lat, lon, target_date)
        if target_date not in {"current", "today", "now"} and self.redis:
            try:
                await self.redis.setex(
                    f"weather:read:{norm_loc}:{target_date}",
                    self.settings.weather_cache_ttl_seconds,
                    fact.model_dump_json(),
                )
            except Exception as exc:
                logger.warning("Forecast cache write failed: %s", exc)
        return fact

    async def _query_verified_data(self, location: str, lat: float, lon: float, target_date: str) -> WeatherDataFact:
        if target_date in {"current", "today", "now"}:
            db_fact = await self._query_database_observation(location, lat, lon)
            if db_fact is not None and db_fact.freshness == "fresh":
                return db_fact
        return await self._fetch_openmeteo(location, lat, lon, target_date)

    async def _query_database_observation(self, location: str, lat: float, lon: float) -> WeatherDataFact | None:
        try:
            async with self.pool.connection() as conn:
                conn.row_factory = dict_row
                cur = await conn.execute(
                    """
                    SELECT location_name, latitude, longitude, observed_at, ingested_at,
                           source, temp_c, humidity_pct, precipitation_mm,
                           wind_speed_kph, weather_code, provenance
                    FROM weather_observations
                    WHERE ST_DWithin(
                        geom,
                        ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326)::geography,
                        20000
                    )
                    ORDER BY observed_at DESC
                    LIMIT 1
                    """,
                    {"lat": lat, "lon": lon},
                )
                record: dict[str, Any] | None = await cur.fetchone()
        except Exception as exc:
            logger.warning("TimescaleDB observation query failed: %s", exc)
            return None
        if not record or record.get("temp_c") is None:
            return None
        observed_at = record.get("observed_at")
        provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
        return WeatherDataFact(
            location=str(record.get("location_name") or location),
            latitude=float(record.get("latitude") or lat),
            longitude=float(record.get("longitude") or lon),
            target_date="current",
            temp_c=float(record["temp_c"]),
            temp_min_c=float(record["temp_c"]),
            temp_max_c=float(record["temp_c"]),
            humidity_pct=float(record["humidity_pct"]) if record.get("humidity_pct") is not None else None,
            precipitation_mm=float(record["precipitation_mm"]) if record.get("precipitation_mm") is not None else None,
            precipitation_probability_pct=None,
            wind_speed_kph=float(record["wind_speed_kph"]) if record.get("wind_speed_kph") is not None else None,
            weather_code=str(record["weather_code"]) if record.get("weather_code") is not None else None,
            condition_description=_wmo_to_condition(record.get("weather_code")),
            will_rain=float(record["precipitation_mm"]) > 0 if record.get("precipitation_mm") is not None else False,
            source=str(record.get("source") or "timescaledb"),
            source_url=provenance.get("source_url"),
            observed_at=observed_at,
            fetched_at=record.get("ingested_at"),
            freshness=_freshness(observed_at, self.settings.weather_freshness_seconds),
        )

    async def _fetch_openmeteo(self, location: str, lat: float, lon: float, target_date: str) -> WeatherDataFact:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weather_code,wind_speed_10m_max",
                    "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m,wind_direction_10m,surface_pressure",
                    "timezone": "auto",
                    "forecast_days": 7,
                },
            )
            response.raise_for_status()
            data = response.json()
        daily = data.get("daily") or {}
        current = data.get("current") or {}
        dates = daily.get("time") or []
        idx = 1 if target_date == "tomorrow" else 0
        if idx >= len(dates):
            raise ValueError(f"Verified forecast is unavailable for {location} on {target_date}")
        fields = {
            "temp_max": (daily.get("temperature_2m_max") or [None] * len(dates))[idx],
            "temp_min": (daily.get("temperature_2m_min") or [None] * len(dates))[idx],
            "precip": (daily.get("precipitation_sum") or [None] * len(dates))[idx],
            "precip_prob": (daily.get("precipitation_probability_max") or [None] * len(dates))[idx],
            "wind": (daily.get("wind_speed_10m_max") or [None] * len(dates))[idx],
            "wcode": (daily.get("weather_code") or [None] * len(dates))[idx],
        }
        if any(value is None for value in fields.values()):
            raise ValueError(f"Verified forecast is incomplete for {location} on {target_date}")
        is_current = target_date in {"current", "today", "now"}
        current_temp = current.get("temperature_2m") if is_current else None
        if is_current and current_temp is None:
            raise ValueError(f"Verified current temperature is unavailable for {location}")
        fetched_at = datetime.now(UTC)
        return WeatherDataFact(
            location=location,
            latitude=lat,
            longitude=lon,
            target_date=target_date,
            temp_c=float(current_temp) if current_temp is not None else round((float(fields["temp_min"]) + float(fields["temp_max"])) / 2.0, 1),
            temp_min_c=float(fields["temp_min"]),
            temp_max_c=float(fields["temp_max"]),
            humidity_pct=float(current["relative_humidity_2m"]) if is_current and current.get("relative_humidity_2m") is not None else None,
            precipitation_mm=float(fields["precip"]),
            precipitation_probability_pct=float(fields["precip_prob"]),
            wind_speed_kph=float(fields["wind"]),
            weather_code=str(fields["wcode"]),
            condition_description=_wmo_to_condition(fields["wcode"]),
            will_rain=float(fields["precip"]) > 0 or float(fields["precip_prob"]) >= 50.0,
            source="open_meteo",
            source_url="https://api.open-meteo.com/v1/forecast",
            observed_at=None,
            fetched_at=fetched_at,
            freshness="fresh",
        )

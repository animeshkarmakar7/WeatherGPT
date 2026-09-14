import json
import logging
import re
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


def _wmo_to_condition(code: str | int) -> str:
    try:
        return _WMO_CONDITIONS.get(int(code), "Unknown")
    except (ValueError, TypeError):
        return "Unknown"


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
            resp = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": norm, "count": 5, "language": "en", "format": "json"},
            )
            resp.raise_for_status()
            results = resp.json().get("results") or []
            if not results:
                raise ValueError(f"No verified location found for '{location}'")
            item = results[0]
            return float(item["latitude"]), float(item["longitude"]), str(item.get("name", location))

    async def get_weather_data(self, location: str, target_date: str = "current") -> WeatherDataFact:
        norm_loc = location.strip().lower()
        cache_key = f"weather:read:{norm_loc}:{target_date}"
        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    data = json.loads(cached)
                    data["cached"] = True
                    return WeatherDataFact.model_validate(data)
            except Exception as exc:
                logger.warning("Weather cache read failed: %s", exc)
        lat, lon, resolved_name = await self._resolve_coordinates(norm_loc)
        fact = await self._query_source(resolved_name, lat, lon, target_date)
        if self.redis:
            try:
                await self.redis.setex(cache_key, self.settings.weather_cache_ttl_seconds, fact.model_dump_json())
            except Exception as exc:
                logger.warning("Weather cache write failed: %s", exc)
        return fact

    async def _query_source(self, location: str, lat: float, lon: float, target_date: str) -> WeatherDataFact:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
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
            resp.raise_for_status()
            data = resp.json()
        daily = data.get("daily") or {}
        current = data.get("current") or {}
        idx = 1 if target_date == "tomorrow" else 0
        dates = daily.get("time") or []
        if idx >= len(dates):
            raise ValueError(f"Verified forecast is unavailable for {location} on {target_date}")
        fields = {
            "temperature_2m_max": daily.get("temperature_2m_max", []),
            "temperature_2m_min": daily.get("temperature_2m_min", []),
            "precipitation_sum": daily.get("precipitation_sum", []),
            "precipitation_probability_max": daily.get("precipitation_probability_max", []),
            "weather_code": daily.get("weather_code", []),
            "wind_speed_10m_max": daily.get("wind_speed_10m_max", []),
        }
        if any(len(values) <= idx or values[idx] is None for values in fields.values()):
            raise ValueError(f"Verified forecast is incomplete for {location} on {target_date}")
        temp_max = float(fields["temperature_2m_max"][idx])
        temp_min = float(fields["temperature_2m_min"][idx])
        precip = float(fields["precipitation_sum"][idx])
        precip_prob = float(fields["precipitation_probability_max"][idx])
        wind = float(fields["wind_speed_10m_max"][idx])
        wcode = str(fields["weather_code"][idx])
        is_current = target_date in {"current", "today", "now"}
        current_temp = current.get("temperature_2m") if is_current else None
        if is_current and current_temp is None:
            raise ValueError(f"Verified current temperature is unavailable for {location}")
        temp = float(current_temp) if current_temp is not None else round((temp_min + temp_max) / 2.0, 1)
        return WeatherDataFact(
            location=location,
            latitude=lat,
            longitude=lon,
            target_date=target_date,
            temp_c=temp,
            temp_min_c=temp_min,
            temp_max_c=temp_max,
            humidity_pct=float(current["relative_humidity_2m"]) if is_current and current.get("relative_humidity_2m") is not None else None,
            precipitation_mm=precip,
            precipitation_probability_pct=precip_prob,
            wind_speed_kph=wind,
            weather_code=wcode,
            condition_description=_wmo_to_condition(wcode),
            will_rain=precip > 0 or precip_prob >= 50.0,
            source="open_meteo",
        )

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any
import httpx
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from redis.asyncio import Redis

from .config import ChatSettings
from .models import WeatherDataFact

logger = logging.getLogger(__name__)

# WMO Weather Interpretation Codes (WW codes) — subset covering common cases.
# Full table: https://open-meteo.com/en/docs#weathervariables
_WMO_CONDITIONS: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _wmo_to_condition(code: str | int, will_rain: bool) -> str:
    """Return a human-readable condition string from a WMO weather code."""
    try:
        return _WMO_CONDITIONS.get(int(code), "Partly cloudy")
    except (ValueError, TypeError):
        return "Rain showers" if will_rain else "Partly cloudy"


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
                row = await cur.fetchone()
                return row is not None
        except Exception:
            return False

    async def _resolve_coordinates(self, location: str) -> tuple[float, float, str]:
        norm = location.strip().lower()
        if norm in self.settings.default_cities:
            lat, lon = self.settings.default_cities[norm]
            return lat, lon, norm

        search_tokens = re.findall(r"\w+", norm)
        candidate_names = [norm] + search_tokens

        async with httpx.AsyncClient(timeout=4.0) as client:
            for name in candidate_names:
                if len(name) < 3:
                    continue
                try:
                    resp = await client.get(
                        "https://geocoding-api.open-meteo.com/v1/search",
                        params={"name": name, "count": 1},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        results = data.get("results")
                        if results and len(results) > 0:
                            item = results[0]
                            return float(item["latitude"]), float(item["longitude"]), item.get("name", location)
                except Exception:
                    pass

        return 18.5204, 73.8567, location

    async def get_weather_data(self, location: str, target_date: str = "tomorrow") -> WeatherDataFact:
        norm_loc = location.strip().lower()
        cache_key = f"weather:read:{norm_loc}:{target_date}"

        if self.redis:
            try:
                cached_json = await self.redis.get(cache_key)
                if cached_json:
                    data = json.loads(cached_json)
                    data["cached"] = True
                    return WeatherDataFact.model_validate(data)
            except Exception as e:
                logger.warning("Redis cache read failed: %s", e)

        lat, lon, resolved_name = await self._resolve_coordinates(norm_loc)
        fact = await self._query_timescaledb(resolved_name, (lat, lon), target_date)

        if self.redis and fact:
            try:
                await self.redis.setex(
                    cache_key,
                    self.settings.weather_cache_ttl_seconds,
                    fact.model_dump_json(),
                )
            except Exception as e:
                logger.warning("Redis cache write failed: %s", e)

        return fact

    async def _query_timescaledb(
        self, location: str, coords: tuple[float, float], target_date: str
    ) -> WeatherDataFact:
        lat, lon = coords
        db_record: dict[str, Any] | None = None

        try:
            async with self.pool.connection() as conn:
                conn.row_factory = dict_row
                cur = await conn.execute(
                    """
                    SELECT *
                    FROM weather_observations
                    WHERE location_name ILIKE %(location)s
                    ORDER BY observed_at DESC
                    LIMIT 1
                    """,
                    {"location": f"%{location.lower()}%"},
                )
                db_record = await cur.fetchone()
        except Exception as exc:
            logger.warning("TimescaleDB query failed (fallback to live API query): %s", exc)

        if db_record and target_date in ("today", "current", "now"):
            precip = db_record.get("precipitation_mm") or 0.0
            temp = db_record.get("temp_c")
            return WeatherDataFact(
                location=location,
                latitude=lat,
                longitude=lon,
                target_date=target_date,
                temp_c=temp,
                temp_min_c=temp,
                temp_max_c=temp,
                humidity_pct=db_record.get("humidity_pct"),
                precipitation_mm=precip,
                precipitation_probability_pct=min(100.0, precip * 25.0) if precip > 0 else 10.0,
                wind_speed_kph=db_record.get("wind_speed_kph"),
                weather_code=str(db_record.get("weather_code") or "1"),
                condition_description="Rainy" if precip > 0.5 else "Partly cloudy",
                will_rain=precip > 0.5,
                source="timescaledb",
            )

        return await self._fetch_openmeteo_forecast(location, lat, lon, target_date)

    async def _fetch_openmeteo_forecast(
        self, location: str, lat: float, lon: float, target_date: str
    ) -> WeatherDataFact:
        async with httpx.AsyncClient(timeout=8.0) as client:
            try:
                resp = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weather_code,wind_speed_10m_max",
                        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
                        "timezone": "auto",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                daily = data.get("daily", {})
                current = data.get("current", {})

                idx = 1 if target_date == "tomorrow" else 0
                dates = daily.get("time", [])
                if len(dates) <= idx:
                    idx = 0

                temp_max = daily.get("temperature_2m_max", [28.0])[idx]
                temp_min = daily.get("temperature_2m_min", [22.0])[idx]
                precip = daily.get("precipitation_sum", [0.0])[idx]
                precip_prob = daily.get("precipitation_probability_max", [20.0])[idx]
                wind = daily.get("wind_speed_10m_max", [12.0])[idx]
                wcode = str(daily.get("weather_code", [1])[idx])

                curr_temp = current.get("temperature_2m")
                avg_temp = curr_temp if (target_date in ("today", "current", "now") and curr_temp is not None) else round((temp_min + temp_max) / 2.0, 1)

                will_rain = bool((precip and precip > 1.0) or (precip_prob and precip_prob >= 50.0))
                condition = _wmo_to_condition(wcode, will_rain)

                return WeatherDataFact(
                    location=location,
                    latitude=lat,
                    longitude=lon,
                    target_date=target_date,
                    temp_min_c=temp_min,
                    temp_max_c=temp_max,
                    temp_c=avg_temp,
                    precipitation_mm=precip,
                    precipitation_probability_pct=float(precip_prob or 0.0),
                    wind_speed_kph=wind,
                    weather_code=wcode,
                    condition_description=condition,
                    will_rain=will_rain,
                    source="open_meteo_forecast",
                )
            except Exception as e:
                logger.error("Forecast API fallback failed: %s", e)
                return WeatherDataFact(
                    location=location,
                    latitude=lat,
                    longitude=lon,
                    target_date=target_date,
                    temp_min_c=21.5,
                    temp_max_c=29.0,
                    temp_c=25.2,
                    precipitation_mm=4.5,
                    precipitation_probability_pct=65.0,
                    wind_speed_kph=14.0,
                    weather_code="61",
                    condition_description="Scattered light rain",
                    will_rain=True,
                    source="weathergpt_baseline",
                )

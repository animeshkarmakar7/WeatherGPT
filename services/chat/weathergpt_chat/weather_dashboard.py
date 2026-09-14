from datetime import UTC, datetime

import httpx

from .models import WeatherDataFact


class WeatherDashboardService:
    def __init__(self, query_service) -> None:
        self.query_service = query_service

    async def get_dashboard(self, latitude: float, longitude: float, location_name: str | None = None) -> dict:
        fact = None
        resolved_name = location_name or "Current location"
        try:
            fact = await self.query_service._query_database_observation(resolved_name, latitude, longitude)
        except Exception:
            fact = None
        if fact is None or fact.freshness != "fresh":
            fact = None
        forecast = await self._fetch_forecast(latitude, longitude)
        current = await self._fetch_current(latitude, longitude) if fact is None else None
        if fact is None and current is None:
            raise RuntimeError("Verified live weather data is unavailable for this location")
        if fact is None and current is not None:
            fact = current
        return {
            "location": resolved_name,
            "latitude": latitude,
            "longitude": longitude,
            "current": fact.model_dump(mode="json"),
            "forecast": forecast,
            "map": {
                "latitude": latitude,
                "longitude": longitude,
                "provider": "OpenStreetMap",
            },
            "generated_at": datetime.now(UTC).isoformat(),
        }

    async def search_locations(self, query: str) -> list[dict]:
        value = query.strip()
        if len(value) < 2:
            return []
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": value, "count": 8, "language": "en", "format": "json", "countryCode": "IN"},
            )
            response.raise_for_status()
            results = response.json().get("results") or []
        return [
            {
                "name": item.get("name"),
                "admin1": item.get("admin1"),
                "country": item.get("country"),
                "country_code": item.get("country_code"),
                "latitude": item.get("latitude"),
                "longitude": item.get("longitude"),
                "timezone": item.get("timezone"),
            }
            for item in results
        ]

    async def _fetch_current(self, latitude: float, longitude: float) -> WeatherDataFact | None:
        data = await self._fetch(latitude, longitude, current=True, forecast_days=1)
        current = data.get("current") or {}
        if current.get("temperature_2m") is None or current.get("time") is None:
            return None
        observed_at = datetime.fromisoformat(str(current["time"]))
        return WeatherDataFact(
            location="Current location",
            latitude=latitude,
            longitude=longitude,
            target_date="current",
            temp_c=float(current["temperature_2m"]),
            humidity_pct=float(current["relative_humidity_2m"]) if current.get("relative_humidity_2m") is not None else None,
            precipitation_mm=float(current.get("precipitation") or 0.0),
            wind_speed_kph=float(current["wind_speed_10m"]) if current.get("wind_speed_10m") is not None else None,
            weather_code=str(current.get("weather_code")) if current.get("weather_code") is not None else None,
            source="open_meteo",
            source_url="https://api.open-meteo.com/v1/forecast",
            observed_at=observed_at,
            fetched_at=datetime.now(UTC),
            freshness="fresh",
        )

    async def _fetch_forecast(self, latitude: float, longitude: float) -> list[dict]:
        data = await self._fetch(latitude, longitude, current=False, forecast_days=10)
        daily = data.get("daily") or {}
        dates = daily.get("time") or []
        result = []
        for index, date in enumerate(dates):
            values = {
                "temperature_max_c": daily.get("temperature_2m_max", [None] * len(dates))[index],
                "temperature_min_c": daily.get("temperature_2m_min", [None] * len(dates))[index],
                "precipitation_mm": daily.get("precipitation_sum", [None] * len(dates))[index],
                "precipitation_probability_pct": daily.get("precipitation_probability_max", [None] * len(dates))[index],
                "wind_speed_kph": daily.get("wind_speed_10m_max", [None] * len(dates))[index],
                "weather_code": daily.get("weather_code", [None] * len(dates))[index],
                "sunrise": daily.get("sunrise", [None] * len(dates))[index],
                "sunset": daily.get("sunset", [None] * len(dates))[index],
            }
            if any(value is None for value in values.values()):
                continue
            result.append({"date": date, **values})
        return result

    async def _fetch(self, latitude: float, longitude: float, current: bool, forecast_days: int) -> dict:
        current_fields = "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m,wind_direction_10m,surface_pressure,time" if current else None
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": current_fields,
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weather_code,wind_speed_10m_max,sunrise,sunset",
                    "timezone": "auto",
                    "forecast_days": forecast_days,
                },
            )
            response.raise_for_status()
            return response.json()

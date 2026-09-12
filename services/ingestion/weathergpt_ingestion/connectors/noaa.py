from datetime import UTC, datetime

from ..config import Settings
from ..models import Provenance, RawWeatherEvent, SourceName
from ..resilience import CircuitOpenError
from ..topics import NOAA_FORECAST
from .base import WeatherConnector


class NoaaForecastConnector(WeatherConnector):
    @property
    def name(self) -> str:
        return "noaa_forecast"

    @property
    def topic(self) -> str:
        return NOAA_FORECAST

    async def fetch_current(self, location_name: str, latitude: float, longitude: float) -> RawWeatherEvent:
        key = self.cache_key(location_name, latitude, longitude)

        async def _fetch() -> RawWeatherEvent:
            points_url = f"{self.settings.noaa_base_url}/points/{latitude:.4f},{longitude:.4f}"
            points_response = await self.client.get(str(points_url), headers={"User-Agent": "WeatherGPT/0.1"})
            points_response.raise_for_status()
            points_payload = points_response.json()
            forecast_url = points_payload["properties"]["forecastHourly"]
            forecast_response = await self.client.get(forecast_url, headers={"User-Agent": "WeatherGPT/0.1"})
            forecast_response.raise_for_status()
            payload = forecast_response.json()
            periods = payload.get("properties", {}).get("periods") or []
            first_period = periods[0] if periods else {}
            observed_at = datetime.fromisoformat(first_period.get("startTime", datetime.now(UTC).isoformat()))
            event = RawWeatherEvent(
                source=SourceName.NOAA,
                topic=self.topic,
                external_id=f"noaa:{latitude:.4f}:{longitude:.4f}:{first_period.get('number', 0)}",
                location_name=location_name,
                latitude=latitude,
                longitude=longitude,
                observed_at=observed_at,
                payload=payload,
                provenance=Provenance(
                    source=SourceName.NOAA,
                    connector=self.name,
                    source_url=forecast_url,
                    license="Public domain NOAA/NWS data; verify downstream attribution requirements.",
                    attribution="NOAA National Weather Service",
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


def build_noaa(settings: Settings, redis):
    return NoaaForecastConnector(settings, redis)

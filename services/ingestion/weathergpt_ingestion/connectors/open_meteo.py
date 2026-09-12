from datetime import datetime

from ..config import Settings
from ..models import Provenance, RawWeatherEvent, SourceName
from ..resilience import CircuitOpenError
from ..topics import OPEN_METEO_CURRENT
from .base import WeatherConnector


class OpenMeteoConnector(WeatherConnector):
    @property
    def name(self) -> str:
        return "open_meteo_current"

    @property
    def topic(self) -> str:
        return OPEN_METEO_CURRENT

    async def fetch_current(self, location_name: str, latitude: float, longitude: float) -> RawWeatherEvent:
        key = self.cache_key(location_name, latitude, longitude)

        async def _fetch() -> RawWeatherEvent:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": ",".join(
                    [
                        "temperature_2m",
                        "relative_humidity_2m",
                        "precipitation",
                        "weather_code",
                        "surface_pressure",
                        "wind_speed_10m",
                        "wind_direction_10m",
                    ]
                ),
                "timezone": "auto",
            }
            url = f"{self.settings.open_meteo_base_url}/v1/forecast"
            response = await self.client.get(str(url), params=params)
            response.raise_for_status()
            payload = response.json()
            current = payload.get("current") or {}
            observed_at = datetime.fromisoformat(str(current["time"]).replace("Z", "+00:00"))
            event = RawWeatherEvent(
                source=SourceName.OPEN_METEO,
                topic=self.topic,
                external_id=f"open-meteo:{latitude:.4f}:{longitude:.4f}:{current['time']}",
                location_name=location_name,
                latitude=latitude,
                longitude=longitude,
                observed_at=observed_at,
                payload=payload,
                provenance=Provenance(
                    source=SourceName.OPEN_METEO,
                    connector=self.name,
                    source_url=str(response.url),
                    license="CC BY 4.0 for most Open-Meteo data sources; verify per upstream source.",
                    attribution="Open-Meteo",
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


def build_open_meteo(settings: Settings, redis):
    return OpenMeteoConnector(settings, redis)

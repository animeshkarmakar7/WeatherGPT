from datetime import UTC, datetime

from ..config import Settings
from ..models import Provenance, RawWeatherEvent, SourceName
from ..topics import IMD_BULLETIN
from .base import WeatherConnector


class ImdBulletinConnector(WeatherConnector):
    @property
    def name(self) -> str:
        return "imd_bulletin"

    @property
    def topic(self) -> str:
        return IMD_BULLETIN

    async def fetch_current(self, location_name: str, latitude: float, longitude: float) -> RawWeatherEvent:
        if self.settings.imd_base_url is None:
            raise RuntimeError("WEATHERGPT_IMD_BASE_URL is required for IMD ingestion")

        response = await self.client.get(str(self.settings.imd_base_url))
        response.raise_for_status()
        payload = {"bulletin_text": response.text}
        return RawWeatherEvent(
            source=SourceName.IMD,
            topic=self.topic,
            external_id=f"imd:{location_name.lower()}:{response.headers.get('last-modified', 'latest')}",
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            observed_at=datetime.now(UTC),
            payload=payload,
            provenance=Provenance(
                source=SourceName.IMD,
                connector=self.name,
                source_url=str(response.url),
                attribution="India Meteorological Department",
            ),
        )

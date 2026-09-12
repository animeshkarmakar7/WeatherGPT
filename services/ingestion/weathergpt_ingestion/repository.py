from typing import Any

import psycopg
from psycopg.types.json import Jsonb
from psycopg.rows import dict_row

from .models import DeadLetterEvent, IngestionRun, IngestionStatus, NormalizedObservation


class WeatherRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url.replace("postgresql+psycopg://", "postgresql://")

    async def save_observation(self, observation: NormalizedObservation) -> None:
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            await conn.execute(
                """
                INSERT INTO weather_observations (
                    observed_at, source, external_id, location_name, latitude, longitude,
                    temp_c, wind_speed_kph, wind_direction_deg, humidity_pct,
                    precipitation_mm, pressure_hpa, weather_code, quality_flags,
                    provenance, raw_payload
                )
                VALUES (
                    %(observed_at)s, %(source)s, %(external_id)s, %(location_name)s,
                    %(latitude)s, %(longitude)s, %(temp_c)s, %(wind_speed_kph)s,
                    %(wind_direction_deg)s, %(humidity_pct)s, %(precipitation_mm)s,
                    %(pressure_hpa)s, %(weather_code)s, %(quality_flags)s,
                    %(provenance)s, %(raw_payload)s
                )
                ON CONFLICT (observed_at, source, external_id) DO UPDATE SET
                    ingested_at = now(),
                    temp_c = EXCLUDED.temp_c,
                    wind_speed_kph = EXCLUDED.wind_speed_kph,
                    wind_direction_deg = EXCLUDED.wind_direction_deg,
                    humidity_pct = EXCLUDED.humidity_pct,
                    precipitation_mm = EXCLUDED.precipitation_mm,
                    pressure_hpa = EXCLUDED.pressure_hpa,
                    weather_code = EXCLUDED.weather_code,
                    quality_flags = EXCLUDED.quality_flags,
                    provenance = EXCLUDED.provenance,
                    raw_payload = EXCLUDED.raw_payload
                """,
                _observation_params(observation),
            )

    async def ping(self) -> bool:
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            cursor = await conn.execute("SELECT 1")
            row = await cursor.fetchone()
            return row is not None

    async def latest_observation(self, city: str) -> dict[str, Any] | None:
        async with await psycopg.AsyncConnection.connect(self.database_url, row_factory=dict_row) as conn:
            cursor = await conn.execute(
                """
                SELECT *
                FROM weather_observations
                WHERE location_name = %(city)s
                ORDER BY observed_at DESC
                LIMIT 1
                """,
                {"city": city.lower()},
            )
            return await cursor.fetchone()

    async def list_runs(self, limit: int = 25) -> list[dict[str, Any]]:
        async with await psycopg.AsyncConnection.connect(self.database_url, row_factory=dict_row) as conn:
            cursor = await conn.execute(
                """
                SELECT *
                FROM ingestion_runs
                ORDER BY started_at DESC
                LIMIT %(limit)s
                """,
                {"limit": limit},
            )
            return list(await cursor.fetchall())

    async def list_dead_letters(self, limit: int = 25) -> list[dict[str, Any]]:
        async with await psycopg.AsyncConnection.connect(self.database_url, row_factory=dict_row) as conn:
            cursor = await conn.execute(
                """
                SELECT *
                FROM ingestion_dead_letters
                ORDER BY created_at DESC
                LIMIT %(limit)s
                """,
                {"limit": limit},
            )
            return list(await cursor.fetchall())

    async def save_dead_letter(self, event: DeadLetterEvent) -> None:
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            await conn.execute(
                """
                INSERT INTO ingestion_dead_letters (
                    id, source, topic, error_type, error_message, payload, created_at
                )
                VALUES (
                    %(id)s, %(source)s, %(topic)s, %(error_type)s,
                    %(error_message)s, %(payload)s, %(created_at)s
                )
                """,
                {
                    "id": event.id,
                    "source": event.source.value,
                    "topic": event.topic,
                    "error_type": event.error_type,
                    "error_message": event.error_message,
                    "payload": Jsonb(event.payload),
                    "created_at": event.created_at,
                },
            )

    async def start_run(self, run: IngestionRun) -> None:
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            await conn.execute(
                """
                INSERT INTO ingestion_runs (id, source, connector, status, started_at)
                VALUES (%(id)s, %(source)s, %(connector)s, %(status)s, %(started_at)s)
                """,
                {
                    "id": run.id,
                    "source": run.source.value,
                    "connector": run.connector,
                    "status": run.status.value,
                    "started_at": run.started_at,
                },
            )

    async def finish_run(
        self,
        run: IngestionRun,
        status: IngestionStatus,
        records_fetched: int,
        records_published: int,
        error_message: str | None = None,
    ) -> None:
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            await conn.execute(
                """
                UPDATE ingestion_runs
                SET status = %(status)s,
                    finished_at = now(),
                    records_fetched = %(records_fetched)s,
                    records_published = %(records_published)s,
                    error_message = %(error_message)s
                WHERE id = %(id)s
                """,
                {
                    "id": run.id,
                    "status": status.value,
                    "records_fetched": records_fetched,
                    "records_published": records_published,
                    "error_message": error_message,
                },
            )


def _observation_params(observation: NormalizedObservation) -> dict[str, Any]:
    data = observation.model_dump(mode="json")
    data["source"] = observation.source.value
    data["quality_flags"] = Jsonb([flag.value for flag in observation.quality_flags])
    data["provenance"] = Jsonb(observation.provenance.model_dump(mode="json"))
    data["raw_payload"] = Jsonb(observation.raw_payload)
    return data

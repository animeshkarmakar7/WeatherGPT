"""Spark Structured Streaming job: weather.normalized.observation.v1 -> TimescaleDB.

Architecture
------------
* Reads from the normalized Kafka topic (Spark is a *secondary* analytics consumer;
  the authoritative Python observation-writer consumer owns the primary write path).
* Each micro-batch is written in two steps:
  1. Stage into weather_observations_stream_staging (hypertable) via JDBC append
     so the raw batch is durable and inspectable for analytics / replay.
  2. Promote staged rows into weather_observations via the
     upsert_weather_observation() stored function, which uses the same
     ON CONFLICT DO UPDATE semantics as the Python writer.
     This makes Kafka replay safe -- a row replayed from any path is an
     idempotent upsert keyed on (observed_at, source, external_id).

Environment variables
---------------------
WEATHERGPT_KAFKA_BOOTSTRAP_SERVERS   e.g. kafka:29092
WEATHERGPT_JDBC_DATABASE_URL         e.g. jdbc:postgresql://timescaledb:5432/weathergpt
WEATHERGPT_DATABASE_USER             default: weathergpt
WEATHERGPT_DATABASE_PASSWORD         default: weathergpt
WEATHERGPT_SPARK_CHECKPOINT_LOCATION default: /checkpoints/kafka-normalized-to-staging
WEATHERGPT_SPARK_TRIGGER_SECONDS     default: 30 (micro-batch interval)
"""

import json
import os

import psycopg
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, from_json, lit
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# Schema for the normalized observation JSON value on the Kafka topic.
# Must stay in sync with NormalizedObservation in models.py.
# ---------------------------------------------------------------------------
NORMALIZED_OBSERVATION_SCHEMA = StructType(
    [
        StructField("observed_at", TimestampType(), nullable=False),
        StructField("source", StringType(), nullable=False),
        StructField("external_id", StringType(), nullable=False),
        StructField("location_name", StringType(), nullable=False),
        StructField("latitude", DoubleType(), nullable=False),
        StructField("longitude", DoubleType(), nullable=False),
        StructField("temp_c", DoubleType(), nullable=True),
        StructField("wind_speed_kph", DoubleType(), nullable=True),
        StructField("wind_direction_deg", DoubleType(), nullable=True),
        StructField("humidity_pct", DoubleType(), nullable=True),
        StructField("precipitation_mm", DoubleType(), nullable=True),
        StructField("pressure_hpa", DoubleType(), nullable=True),
        StructField("weather_code", StringType(), nullable=True),
        # quality_flags and provenance arrive as JSON strings in the Kafka
        # message; we keep them as StringType here and cast to JSONB in the
        # stored function.  Spark JDBC does not natively map to JSONB.
        StructField("quality_flags", StringType(), nullable=True),
        StructField("provenance", StringType(), nullable=True),
        StructField("raw_payload", StringType(), nullable=True),
    ]
)

# Columns that the staging table expects (mirrors weather_observations minus geom).
_STAGING_COLUMNS = [
    "observed_at",
    "source",
    "external_id",
    "location_name",
    "latitude",
    "longitude",
    "temp_c",
    "wind_speed_kph",
    "wind_direction_deg",
    "humidity_pct",
    "precipitation_mm",
    "pressure_hpa",
    "weather_code",
    "quality_flags",
    "provenance",
    "raw_payload",
]


def _jdbc_properties(user: str, password: str) -> dict:
    return {
        "user": user,
        "password": password,
        "driver": "org.postgresql.Driver",
        "batchsize": "1000",
        "reWriteBatchedInserts": "true",
    }


def _upsert_batch_to_observations(batch_df: DataFrame, database_url: str, user: str, password: str) -> None:
    """Promote a micro-batch into weather_observations via the stored upsert function.

    We use psycopg (Python) rather than Spark JDBC for the upsert call because:
    - JDBC `.mode("append")` emits plain INSERT statements that violate the PK
      on Kafka replay.
    - Spark has no built-in JDBC ON CONFLICT support.
    - The stored function upsert_weather_observation() accepts JSONB parameters
      for quality_flags / provenance / raw_payload, which JDBC cannot bind
      directly.

    This runs on the *driver* node.  For very high-volume production use,
    replace with a Spark-side foreachPartition that opens one connection per
    executor, but for Phase 1 the driver-side batch is sufficient.
    """
    rows = batch_df.collect()
    if not rows:
        return

    # Convert JDBC URL to psycopg conninfo (strip "jdbc:" prefix).
    # e.g. jdbc:postgresql://timescaledb:5432/weathergpt
    #   -> postgresql://weathergpt:secret@timescaledb:5432/weathergpt
    pg_dsn = database_url.replace("jdbc:", "", 1)
    pg_dsn = pg_dsn.replace("postgresql://", f"postgresql://{user}:{password}@", 1)

    upsert_sql = """
        SELECT upsert_weather_observation(
            %s::timestamptz, %s, %s, %s,
            %s::double precision, %s::double precision,
            %s::double precision, %s::double precision, %s::double precision,
            %s::double precision, %s::double precision, %s::double precision,
            %s,
            %s::jsonb, %s::jsonb, %s::jsonb
        )
    """

    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            params_list = []
            for row in rows:
                quality_flags = row.quality_flags or "[]"
                provenance = row.provenance or "{}"
                raw_payload = row.raw_payload or "{}"
                # Ensure JSON strings are valid; fall back to safe defaults.
                for field, default in [
                    ("quality_flags", "[]"),
                    ("provenance", "{}"),
                    ("raw_payload", "{}"),
                ]:
                    val = getattr(row, field, None) or default
                    try:
                        json.loads(val)
                    except (TypeError, ValueError):
                        val = default
                    if field == "quality_flags":
                        quality_flags = val
                    elif field == "provenance":
                        provenance = val
                    else:
                        raw_payload = val

                params_list.append((
                    row.observed_at,
                    row.source,
                    row.external_id,
                    row.location_name,
                    row.latitude,
                    row.longitude,
                    row.temp_c,
                    row.wind_speed_kph,
                    row.wind_direction_deg,
                    row.humidity_pct,
                    row.precipitation_mm,
                    row.pressure_hpa,
                    row.weather_code,
                    quality_flags,
                    provenance,
                    raw_payload,
                ))
            cur.executemany(upsert_sql, params_list)
        conn.commit()


def write_batch(
    batch_df: DataFrame,
    batch_id: int,
    database_url: str,
    user: str,
    password: str,
) -> None:
    """foreachBatch handler: stage then upsert into the authoritative table."""
    if batch_df.rdd.isEmpty():
        return

    # Step 1 — Write raw batch into the staging hypertable for analytics /
    # replay inspection.  ON CONFLICT is safe because staging has the same PK.
    staging_df = batch_df.select(*_STAGING_COLUMNS)
    (
        staging_df.write.format("jdbc")
        .option("url", database_url)
        .option("dbtable", "weather_observations_stream_staging")
        .option("user", user)
        .option("password", password)
        .option("driver", "org.postgresql.Driver")
        .option("batchsize", 1000)
        .option("reWriteBatchedInserts", "true")
        # Use INSERT ... ON CONFLICT DO NOTHING via a custom query so staging
        # writes are idempotent on Kafka replay.
        .option(
            "insertStatement",
            (
                "INSERT INTO weather_observations_stream_staging "
                "(observed_at,source,external_id,location_name,latitude,longitude,"
                "temp_c,wind_speed_kph,wind_direction_deg,humidity_pct,"
                "precipitation_mm,pressure_hpa,weather_code,quality_flags,"
                "provenance,raw_payload) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?::jsonb,?::jsonb,?::jsonb) "
                "ON CONFLICT (observed_at,source,external_id) DO NOTHING"
            ),
        )
        .mode("append")
        .save()
    )

    # Step 2 — Upsert into weather_observations (the authoritative read table)
    # via the stored function so the exit-criteria query works.
    _upsert_batch_to_observations(batch_df, database_url, user, password)


def main() -> None:
    kafka_bootstrap = os.environ.get("WEATHERGPT_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    database_url = os.environ["WEATHERGPT_JDBC_DATABASE_URL"]
    database_user = os.environ.get("WEATHERGPT_DATABASE_USER", "weathergpt")
    database_password = os.environ.get("WEATHERGPT_DATABASE_PASSWORD", "weathergpt")
    checkpoint_location = os.environ.get(
        "WEATHERGPT_SPARK_CHECKPOINT_LOCATION",
        "/checkpoints/kafka-normalized-to-staging",
    )
    trigger_seconds = int(os.environ.get("WEATHERGPT_SPARK_TRIGGER_SECONDS", "30"))

    spark = (
        SparkSession.builder.appName("weathergpt-kafka-normalized-to-timescale")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

    normalized = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap)
        .option("subscribe", "weather.normalized.observation.v1")
        .option("startingOffsets", "latest")
        # Spark is a secondary analytics consumer; cap fetch to avoid
        # overwhelming the broker when catching up after a restart.
        .option("maxOffsetsPerTrigger", 10000)
        .load()
    )

    parsed = (
        normalized.select(
            from_json(col("value").cast("string"), NORMALIZED_OBSERVATION_SCHEMA).alias("event")
        )
        .select("event.*")
        # Drop rows where required fields parsed as null (malformed JSON or
        # schema mismatch); these are handled by the Python observation-writer
        # DLQ path and do not need a second DLQ in Spark for Phase 1.
        .filter(col("source").isNotNull() & col("external_id").isNotNull())
    )

    (
        parsed.writeStream.foreachBatch(
            lambda df, bid: write_batch(df, bid, database_url, database_user, database_password)
        )
        .option("checkpointLocation", checkpoint_location)
        .trigger(processingTime=f"{trigger_seconds} seconds")
        .start()
        .awaitTermination()
    )


if __name__ == "__main__":
    main()

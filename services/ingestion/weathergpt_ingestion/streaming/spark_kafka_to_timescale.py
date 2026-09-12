import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import ArrayType, DoubleType, StringType, StructField, StructType, TimestampType

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
        StructField("quality_flags", ArrayType(StringType()), nullable=True),
        StructField("provenance", StringType(), nullable=True),
        StructField("raw_payload", StringType(), nullable=True),
    ]
)


def main() -> None:
    kafka_bootstrap = os.environ.get("WEATHERGPT_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    database_url = os.environ["WEATHERGPT_JDBC_DATABASE_URL"]
    database_user = os.environ.get("WEATHERGPT_DATABASE_USER", "weathergpt")
    database_password = os.environ.get("WEATHERGPT_DATABASE_PASSWORD", "weathergpt")
    checkpoint_location = os.environ.get(
        "WEATHERGPT_SPARK_CHECKPOINT_LOCATION",
        "/checkpoints/kafka-normalized-to-staging",
    )

    spark = (
        SparkSession.builder.appName("weathergpt-kafka-normalized-to-staging")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

    normalized = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap)
        .option("subscribe", "weather.normalized.observation.v1")
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = (
        normalized.select(from_json(col("value").cast("string"), NORMALIZED_OBSERVATION_SCHEMA).alias("event"))
        .select("event.*")
    )

    def write_batch(batch_df, batch_id: int) -> None:
        if batch_df.rdd.isEmpty():
            return
        (
            batch_df.write.format("jdbc")
            .option("url", database_url)
            .option("dbtable", "weather_observations_stream_staging")
            .option("user", database_user)
            .option("password", database_password)
            .option("batchsize", 1000)
            .mode("append")
            .save()
        )

    (
        parsed.writeStream.foreachBatch(write_batch)
        .option("checkpointLocation", checkpoint_location)
        .start()
        .awaitTermination()
    )


if __name__ == "__main__":
    main()

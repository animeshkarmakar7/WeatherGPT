import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import ArrayType, DoubleType, StringType, StructField, StructType, TimestampType

RAW_EVENT_SCHEMA = StructType(
    [
        StructField("source", StringType(), nullable=False),
        StructField("external_id", StringType(), nullable=False),
        StructField("location_name", StringType(), nullable=False),
        StructField("latitude", DoubleType(), nullable=False),
        StructField("longitude", DoubleType(), nullable=False),
        StructField("observed_at", TimestampType(), nullable=False),
        StructField("payload", StringType(), nullable=True),
        StructField("quality_flags", ArrayType(StringType()), nullable=True),
    ]
)


def main() -> None:
    kafka_bootstrap = os.environ.get("WEATHERGPT_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    database_url = os.environ["WEATHERGPT_JDBC_DATABASE_URL"]
    database_user = os.environ.get("WEATHERGPT_DATABASE_USER", "weathergpt")
    database_password = os.environ.get("WEATHERGPT_DATABASE_PASSWORD", "weathergpt")

    spark = (
        SparkSession.builder.appName("weathergpt-kafka-to-timescale")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap)
        .option("subscribePattern", "weather.raw.*.v1")
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = raw.select(from_json(col("value").cast("string"), RAW_EVENT_SCHEMA).alias("event")).select("event.*")

    def write_batch(batch_df, batch_id: int) -> None:
        if batch_df.rdd.isEmpty():
            return
        (
            batch_df.write.format("jdbc")
            .option("url", database_url)
            .option("dbtable", "weather_observations_stream_staging")
            .option("user", database_user)
            .option("password", database_password)
            .mode("append")
            .save()
        )

    (
        parsed.writeStream.foreachBatch(write_batch)
        .option("checkpointLocation", "/tmp/weathergpt/spark/kafka-to-timescale")
        .start()
        .awaitTermination()
    )


if __name__ == "__main__":
    main()

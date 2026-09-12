$ErrorActionPreference = "Stop"

$topics = @(
  @{ Name = "weather.raw.open_meteo.current.v1"; Partitions = 6 },
  @{ Name = "weather.raw.noaa.forecast.v1"; Partitions = 6 },
  @{ Name = "weather.raw.imd.bulletin.v1"; Partitions = 3 },
  @{ Name = "weather.raw.wis2.notification.v1"; Partitions = 6 },
  @{ Name = "weather.normalized.observation.v1"; Partitions = 12 },
  @{ Name = "weather.dlq.ingestion.v1"; Partitions = 3 }
)

$bootstrap = if ($env:KAFKA_BOOTSTRAP_SERVERS) { $env:KAFKA_BOOTSTRAP_SERVERS } else { "localhost:9092" }

foreach ($topic in $topics) {
  kafka-topics `
    --bootstrap-server $bootstrap `
    --create `
    --if-not-exists `
    --topic $topic.Name `
    --partitions $topic.Partitions `
    --replication-factor 1
}

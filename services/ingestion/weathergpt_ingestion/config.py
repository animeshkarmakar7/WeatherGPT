from functools import lru_cache

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WEATHERGPT_", env_file=".env", extra="ignore")

    kafka_bootstrap_servers: str = "localhost:9092"
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql+psycopg://weathergpt:weathergpt@localhost:5432/weathergpt"

    open_meteo_base_url: AnyHttpUrl = "https://api.open-meteo.com"
    noaa_base_url: AnyHttpUrl = "https://api.weather.gov"
    imd_base_url: AnyHttpUrl | None = None
    wis2_mqtt_host: str | None = None
    wis2_mqtt_port: int = 8883
    wis2_mqtt_username: str | None = None
    wis2_mqtt_password: str | None = None
    wis2_mqtt_topic: str = "origin/a/wis2/#"

    request_timeout_seconds: float = 8.0
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_recovery_seconds: int = 60
    last_known_good_ttl_seconds: int = 3600

    default_cities: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {
            "mumbai": (19.0760, 72.8777),
            "pune": (18.5204, 73.8567),
            "delhi": (28.6139, 77.2090),
            "kolkata": (22.5726, 88.3639),
            "chennai": (13.0827, 80.2707),
            "bengaluru": (12.9716, 77.5946),
        }
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

from functools import lru_cache
from pydantic import AnyHttpUrl, Field, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WEATHERGPT_", env_file=".env", extra="ignore")

    # Service & Network
    service_port: int = 8082
    cors_origins: list[str] = ["*"]

    # Storage & Persistence (CQRS Read models)
    database_url: str = "postgresql+psycopg://weathergpt:weathergpt@localhost:5432/weathergpt"
    database_pool_min_size: PositiveInt = 2
    database_pool_max_size: PositiveInt = 10
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = 86400  # 24h
    weather_cache_ttl_seconds: int = 600  # 10 min

    # LLM / vLLM Serving
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "dummy-vllm-key"
    llm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm_temperature: float = 0.1
    llm_timeout_seconds: float = 15.0

    # Geospatial / Location Defaults
    default_cities: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {
            "pune": (18.5204, 73.8567),
            "mumbai": (19.0760, 72.8777),
            "delhi": (28.6139, 77.2090),
            "kolkata": (22.5726, 88.3639),
            "chennai": (13.0827, 80.2707),
            "bengaluru": (12.9716, 77.5946),
        }
    )


@lru_cache
def get_chat_settings() -> ChatSettings:
    return ChatSettings()

from functools import lru_cache
from pydantic import AnyHttpUrl, Field, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WEATHERGPT_", env_file=".env", extra="ignore")

    service_port: int = 8082
    cors_origins: list[str] = ["*"]

    database_url: str = "postgresql+psycopg://weathergpt:weathergpt@localhost:5432/weathergpt"
    database_pool_min_size: PositiveInt = 2
    database_pool_max_size: PositiveInt = 10
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = 86400
    weather_cache_ttl_seconds: int = 600

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "weathergpt"
    minio_secret_key: str = "weathergpt-secret"
    minio_secure: bool = False
    minio_bucket: str = "weathergpt-documents"

    qdrant_url: str = "http://localhost:6333"
    # Set WEATHERGPT_USE_QDRANT=true to activate Qdrant-backed vector store.
    # Defaults to in-memory for tests and lightweight local runs.
    use_qdrant: bool = False

    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "dummy-vllm-key"
    llm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm_temperature: float = 0.1
    llm_timeout_seconds: float = 15.0

    # Set WEATHERGPT_USE_REAL_EMBEDDER=true to load actual BAAI/BGE-M3.
    # Requires FlagEmbedding installed and a GPU/large-CPU instance.
    use_real_embedder: bool = False

    default_cities: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {
            # Tier-1
            "pune": (18.5204, 73.8567),
            "mumbai": (19.0760, 72.8777),
            "delhi": (28.6139, 77.2090),
            "kolkata": (22.5726, 88.3639),
            "chennai": (13.0827, 80.2707),
            "bengaluru": (12.9716, 77.5946),
            # Tier-2
            "nagpur": (21.1458, 79.0882),
            "hyderabad": (17.3850, 78.4867),
            "ahmedabad": (23.0225, 72.5714),
            "jaipur": (26.9124, 75.7873),
            "lucknow": (26.8467, 80.9462),
            "surat": (21.1702, 72.8311),
            "kalyan": (19.2403, 73.1305),
            "dombivli": (19.2183, 73.0881),
            "kalyan dombivli": (19.2403, 73.1305),
            "thane": (19.2183, 72.9781),
            "navi mumbai": (19.0368, 73.0158),
            "aurangabad": (19.8762, 75.3433),
            "nashik": (19.9975, 73.7898),
            "coimbatore": (11.0168, 76.9558),
            "bhopal": (23.2599, 77.4126),
            "visakhapatnam": (17.6868, 83.2185),
            "patna": (25.5941, 85.1376),
            "kochi": (9.9312, 76.2673),
        }
    )


@lru_cache
def get_chat_settings() -> ChatSettings:
    return ChatSettings()


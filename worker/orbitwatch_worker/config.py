from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def env(*names: str):
    return AliasChoices(*names)


class Settings(BaseSettings):
    """Read from worker/.env (git-ignored) or the process environment."""

    model_config = SettingsConfigDict(env_prefix="ORBITWATCH_", env_file=".env", extra="ignore")

    # Temporal (Cloud or a local dev server). With an API key, TLS is switched on automatically.
    temporal_address: str = Field("localhost:7233", validation_alias=env("TEMPORAL_ADDRESS"))
    temporal_namespace: str = Field("default", validation_alias=env("TEMPORAL_NAMESPACE"))
    temporal_api_key: str | None = Field(None, validation_alias=env("TEMPORAL_API_KEY"))
    task_queue: str = Field("orbitwatch", validation_alias=env("TEMPORAL_TASK_QUEUE"))

    api_url: str = "http://localhost:8000"
    worker_token: str | None = None  # must match the server's ORBITWATCH_WORKER_TOKEN

    # Same free public Sentinel-2 catalogue the demo pipeline uses.
    stac_url: str = "https://earth-search.aws.element84.com/v1"
    stac_collection: str = "sentinel-2-c1-l2a"
    max_cloud_cover: float = 40.0


settings = Settings()

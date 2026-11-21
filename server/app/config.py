from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

SERVER_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ORBITWATCH_", env_file=".env", extra="ignore")

    # Folder exported by the Colab demo pipeline (MyDrive/orbitwatch/demo).
    data_dir: Path = SERVER_DIR / "data" / "demo"
    # Where the weekly worker's alerts are kept, and the shared secret it must send to add one.
    alerts_file: Path = SERVER_DIR / "data" / "alerts.json"
    worker_token: str | None = None
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]


settings = Settings()

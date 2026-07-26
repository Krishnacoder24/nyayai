from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # ignore unknown env vars instead of raising an error
    )

    # -------------------------
    # Paths
    # -------------------------

    root_dir: Path = ROOT_DIR

    checkpoint_dir: Path = ROOT_DIR / "model" / "checkpoint"
    corpus_sources_dir: Path = ROOT_DIR / "corpus" / "sources"

    uploads_dir: Path = ROOT_DIR / "data" / "uploads"
    outputs_dir: Path = ROOT_DIR / "data" / "outputs"
    cache_dir: Path = ROOT_DIR / "data" / "cache"
    temp_dir: Path = ROOT_DIR / "data" / "temp"
    celery_broker_url: str = "filesystem://"
    celery_broker_data_folder: str = str(ROOT_DIR / "data" / "celery" / "broker")
    celery_result_backend: str = f"db+sqlite:///{ROOT_DIR / 'data' / 'celery' / 'results.sqlite'}"

    # -------------------------
    # Models
    # -------------------------

    bert_checkpoint: str = "law-ai/InLegalBERT"
    spacy_model: str = "en_core_web_sm"

    # -------------------------
    # Surya
    # -------------------------

    recognition_batch_size: int = Field(
        default=32,
        alias="RECOGNITION_BATCH_SIZE",
    )

    detector_batch_size: int = Field(
        default=4,
        alias="DETECTOR_BATCH_SIZE",
    )

    torch_device: str = Field(
        default="cuda",
        alias="TORCH_DEVICE",
    )

    # -------------------------
    # Infrastructure
    # -------------------------

    qdrant_url: str = Field(
        default="http://localhost:6333",
        alias="QDRANT_URL",
    )

    qdrant_collection: str = Field(
        default="legal_corpus",
        alias="QDRANT_COLLECTION",
    )

    debug: bool = Field(
        default=True,
        alias="DEBUG",
    )

    # -------------------------
    # Housekeeping
    # -------------------------

    output_retention_days: int = Field(
        default=30,
        alias="OUTPUT_RETENTION_DAYS",
    )


settings = Settings()
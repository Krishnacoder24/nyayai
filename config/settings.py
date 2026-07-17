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
    # celery_uploads_dir: str = str(ROOT_DIR / "data" / "uploads")
    # celery_outputs_dir: str = str(ROOT_DIR / "data" / "outputs") 

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

    redis_url: str = Field(
        default="redis://localhost:6379/0",
        alias="REDIS_URL",
    )

    debug: bool = Field(
        default=True,
        alias="DEBUG",
    )


settings = Settings()

# --- add to config/settings.py ---
#
# IMPORTANT: anchor every directory/file setting to an absolute BASE_DIR,
# not a relative path. The API process and the Celery worker process are
# launched separately and won't reliably share a working directory - a
# relative path resolves differently per-process and silently points at
# two different physical directories. This is not a hypothetical: I hit
# it directly - a task sat unclaimed forever with no error, because the
# worker (launched from a different cwd) was watching an entirely
# different folder than the one the API process wrote the task into.
#
# no Redis: Celery's broker is the filesystem transport (messages are
# just files under data/celery/broker/), and the result backend is
# SQLite via SQLAlchemy. both are local files, nothing to run as a
# separate service. one-line swap to redis:// or amqp:// later if this
# ever needs to scale past one machine - nothing in workers/ or api/
# depends on which broker is configured here.

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # adjust .parent count to your actual settings.py depth

# add these fields to your existing Settings(BaseSettings) class:
#
# celery_broker_url: str = "filesystem://"
# celery_broker_data_folder: str = str(BASE_DIR / "data" / "celery" / "broker")
# celery_result_backend: str = f"db+sqlite:///{BASE_DIR / 'data' / 'celery' / 'results.sqlite'}"
# uploads_dir: str = str(BASE_DIR / "data" / "uploads")
# outputs_dir: str = str(BASE_DIR / "data" / "outputs")
"""Celery application setup."""
"""
Celery app - deliberately no Redis.

broker: the filesystem transport. Celery/Kombu ship this built-in - a task
being enqueued is just a file written into celery_broker_data_folder/out,
and the worker process picks it up from there. no separate service to run.

result backend: SQLite via SQLAlchemy (the "db+sqlite:///..." URL format).
also just a local file.

both are one-line swaps to redis:// or amqp:// later if this ever needs to
scale past one machine - nothing in tasks.py or api/ depends on which
broker is configured here.
"""

from pathlib import Path

from celery import Celery

from config.settings import settings
from workers.queues import TASK_ROUTES

# neither the filesystem broker's directories nor the sqlite backend's
# parent directory get created automatically - kombu and sqlite3 both
# error out if they're missing, so this has to happen before anything
# tries to enqueue or connect
_broker_root = Path(settings.celery_broker_data_folder)
(_broker_root / "out").mkdir(parents=True, exist_ok=True)
(_broker_root / "processed").mkdir(parents=True, exist_ok=True)

_sqlite_path = settings.celery_result_backend.removeprefix("db+sqlite:///")
Path(_sqlite_path).parent.mkdir(parents=True, exist_ok=True)

app = Celery("nyayai")

app.conf.broker_url = settings.celery_broker_url #type: ignore
app.conf.broker_transport_options = {
    # producer and consumer run on the same machine here, so "in" and "out"
    # both point at the same directory - see Kombu's filesystem transport docs
    "data_folder_in": f"{settings.celery_broker_data_folder}/out",
    "data_folder_out": f"{settings.celery_broker_data_folder}/out",
    "data_folder_processed": f"{settings.celery_broker_data_folder}/processed",
}
app.conf.result_backend = settings.celery_result_backend #type: ignore
app.conf.task_routes = TASK_ROUTES

# task results (the report dict) are small JSON, not files - fine to keep
# in the sqlite backend indefinitely for now. revisit if this grows.
app.conf.result_expires = None

# app.autodiscover_tasks(["workers"])
app.conf.imports = ("workers.tasks",)
"""
shared dependencies - just enough to avoid every route re-importing the
same things. no database session here yet since there's no database
beyond the celery result backend and the filesystem.
"""

from config.settings import settings
from workers.celery_app import app as celery_app


def get_settings():
    return settings


def get_celery_app():
    return celery_app
"""HTTP error schemas."""
from pydantic import BaseModel


class HTTPError(BaseModel):
    detail: str
from datetime import datetime
from typing import List

from pydantic import BaseModel, HttpUrl


class CodeUpdate(BaseModel):
    url: HttpUrl


class CodeOut(BaseModel):
    short_code: str
    destination_url: str
    short_url: str
    qr_url: str
    created_at: datetime
    scan_count: int
    has_photo: bool
    accent_color: str
    background_color: str


class StatsOut(BaseModel):
    short_code: str
    destination_url: str
    scan_count: int
    recent_scans: List[datetime]

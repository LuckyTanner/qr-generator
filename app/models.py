from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class QRCode(Base):
    __tablename__ = "qr_codes"

    id = Column(Integer, primary_key=True, index=True)
    short_code = Column(String, unique=True, index=True, nullable=False)
    destination_url = Column(String, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    scan_count = Column(Integer, default=0, nullable=False)

    scans = relationship(
        "ScanLog", back_populates="qr_code", cascade="all, delete-orphan"
    )


class ScanLog(Base):
    __tablename__ = "scan_logs"

    id = Column(Integer, primary_key=True, index=True)
    qr_code_id = Column(Integer, ForeignKey("qr_codes.id"), nullable=False)
    scanned_at = Column(DateTime, default=utcnow, nullable=False)

    qr_code = relationship("QRCode", back_populates="scans")

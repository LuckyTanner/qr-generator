import io
import pathlib
import secrets
import string

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import qrcode
from sqlalchemy import desc
from sqlalchemy.orm import Session

from . import models, schemas
from .database import Base, SessionLocal, engine

Base.metadata.create_all(bind=engine)

STATIC_DIR = pathlib.Path(__file__).parent / "static"
ALPHABET = string.ascii_letters + string.digits

app = FastAPI(title="Dynamic QR Code Generator")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def generate_short_code(db: Session, length: int = 7) -> str:
    for _ in range(10):
        code = "".join(secrets.choice(ALPHABET) for _ in range(length))
        exists = db.query(models.QRCode).filter(models.QRCode.short_code == code).first()
        if not exists:
            return code
    raise RuntimeError("Could not generate a unique short code, try again")


def get_code_or_404(db: Session, short_code: str) -> models.QRCode:
    qr_code = db.query(models.QRCode).filter(models.QRCode.short_code == short_code).first()
    if not qr_code:
        raise HTTPException(status_code=404, detail="Short code not found")
    return qr_code


def to_code_out(qr_code: models.QRCode, base_url: str) -> schemas.CodeOut:
    return schemas.CodeOut(
        short_code=qr_code.short_code,
        destination_url=qr_code.destination_url,
        short_url=f"{base_url}r/{qr_code.short_code}",
        created_at=qr_code.created_at,
        scan_count=qr_code.scan_count,
    )


@app.post("/codes")
def create_code(payload: schemas.CodeCreate, request: Request, db: Session = Depends(get_db)):
    short_code = generate_short_code(db)
    qr_code = models.QRCode(short_code=short_code, destination_url=str(payload.url))
    db.add(qr_code)
    db.commit()
    db.refresh(qr_code)

    short_url = f"{request.base_url}r/{short_code}"

    img = qrcode.make(short_url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    headers = {
        "X-Short-Code": short_code,
        "X-Short-Url": short_url,
        "X-Destination-Url": qr_code.destination_url,
    }
    return StreamingResponse(buf, media_type="image/png", headers=headers)


@app.get("/codes", response_model=list[schemas.CodeOut])
def list_codes(request: Request, db: Session = Depends(get_db)):
    codes = db.query(models.QRCode).order_by(desc(models.QRCode.created_at)).all()
    base_url = str(request.base_url)
    return [to_code_out(c, base_url) for c in codes]


@app.put("/codes/{short_code}", response_model=schemas.CodeOut)
def update_code(
    short_code: str,
    payload: schemas.CodeUpdate,
    request: Request,
    db: Session = Depends(get_db),
):
    qr_code = get_code_or_404(db, short_code)
    qr_code.destination_url = str(payload.url)
    db.commit()
    db.refresh(qr_code)
    return to_code_out(qr_code, str(request.base_url))


@app.get("/codes/{short_code}/stats", response_model=schemas.StatsOut)
def get_stats(short_code: str, db: Session = Depends(get_db)):
    qr_code = get_code_or_404(db, short_code)
    recent = (
        db.query(models.ScanLog)
        .filter(models.ScanLog.qr_code_id == qr_code.id)
        .order_by(desc(models.ScanLog.scanned_at))
        .limit(20)
        .all()
    )
    return schemas.StatsOut(
        short_code=qr_code.short_code,
        destination_url=qr_code.destination_url,
        scan_count=qr_code.scan_count,
        recent_scans=[s.scanned_at for s in recent],
    )


@app.get("/r/{short_code}")
def redirect_code(short_code: str, db: Session = Depends(get_db)):
    qr_code = get_code_or_404(db, short_code)
    qr_code.scan_count += 1
    db.add(models.ScanLog(qr_code_id=qr_code.id))
    db.commit()
    return RedirectResponse(url=qr_code.destination_url)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")

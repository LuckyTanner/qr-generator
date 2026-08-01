import io
import pathlib
import secrets
import string
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import HttpUrl, TypeAdapter, ValidationError
from sqlalchemy import desc
from sqlalchemy.orm import Session

from . import models, schemas
from .database import Base, SessionLocal, engine
from .qr_style import DEFAULT_ACCENT, DEFAULT_BACKGROUND, build_qr_image

Base.metadata.create_all(bind=engine)

STATIC_DIR = pathlib.Path(__file__).parent / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

ALPHABET = string.ascii_letters + string.digits
MAX_PHOTO_BYTES = 8 * 1024 * 1024
ALLOWED_PHOTO_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
URL_ADAPTER = TypeAdapter(HttpUrl)

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
        qr_url=f"{base_url}codes/{qr_code.short_code}/qr",
        created_at=qr_code.created_at,
        scan_count=qr_code.scan_count,
        has_photo=bool(qr_code.photo_filename),
        accent_color=qr_code.accent_color or DEFAULT_ACCENT,
        background_color=qr_code.background_color or DEFAULT_BACKGROUND,
    )


def _validate_url(url: str) -> str:
    try:
        return str(URL_ADAPTER.validate_python(url))
    except ValidationError:
        raise HTTPException(status_code=422, detail="Invalid URL")


async def _load_photo(photo: Optional[UploadFile]) -> tuple[Optional[bytes], Optional[str]]:
    """Read + validate an uploaded photo. Returns (bytes, extension) or (None, None)."""
    if photo is None or not photo.filename:
        return None, None

    content = await photo.read()
    if len(content) > MAX_PHOTO_BYTES:
        raise HTTPException(status_code=413, detail="Photo too large (max 8MB)")

    try:
        img = Image.open(io.BytesIO(content))
        img.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    ext = pathlib.Path(photo.filename).suffix.lower()
    if ext not in ALLOWED_PHOTO_EXTENSIONS:
        ext = ".png"
    return content, ext


@app.post("/codes", response_model=None)
async def create_code(
    request: Request,
    url: str = Form(...),
    accent_color: Optional[str] = Form(None),
    background_color: Optional[str] = Form(None),
    photo: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    destination_url = _validate_url(url)
    photo_bytes, photo_ext = await _load_photo(photo)

    short_code = generate_short_code(db)

    photo_filename = None
    if photo_bytes is not None:
        photo_filename = f"{short_code}{photo_ext}"
        (UPLOADS_DIR / photo_filename).write_bytes(photo_bytes)

    qr_code = models.QRCode(
        short_code=short_code,
        destination_url=destination_url,
        photo_filename=photo_filename,
        accent_color=accent_color,
        background_color=background_color,
    )
    db.add(qr_code)
    db.commit()
    db.refresh(qr_code)

    short_url = f"{request.base_url}r/{short_code}"
    pil_photo = Image.open(io.BytesIO(photo_bytes)) if photo_bytes is not None else None
    img = build_qr_image(
        short_url,
        photo=pil_photo,
        accent_color=accent_color,
        background_color=background_color,
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    headers = {
        "X-Short-Code": short_code,
        "X-Short-Url": short_url,
        "X-Destination-Url": qr_code.destination_url,
        "X-Has-Photo": "true" if photo_filename else "false",
    }
    return StreamingResponse(buf, media_type="image/png", headers=headers)


@app.get("/codes/{short_code}/qr")
def get_qr_image(short_code: str, request: Request, db: Session = Depends(get_db)):
    qr_code = get_code_or_404(db, short_code)
    short_url = f"{request.base_url}r/{short_code}"

    pil_photo = None
    if qr_code.photo_filename:
        photo_path = UPLOADS_DIR / qr_code.photo_filename
        if photo_path.exists():
            pil_photo = Image.open(photo_path)
            pil_photo.load()

    img = build_qr_image(
        short_url,
        photo=pil_photo,
        accent_color=qr_code.accent_color,
        background_color=qr_code.background_color,
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


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

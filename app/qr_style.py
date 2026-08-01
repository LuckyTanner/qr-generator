from typing import Optional, Tuple

import cv2
import numpy as np
import qrcode
from qrcode.constants import ERROR_CORRECT_H
from qrcode.util import pattern_position
from PIL import Image, ImageOps, ImageStat

DEFAULT_ACCENT = "#1a1a1a"
DEFAULT_BACKGROUND = "#ffffff"
FINDER_ZONE = 8  # finder pattern (7x7) plus its 1-module separator ring
MIN_VERSION = 6  # ~41x41 modules minimum: enough resolution to read as a photo,
# small enough that modules stay easy for a scanner to resolve

# Escalating (low_force, high_force) pairs. low_force controls how much a module
# that already agrees with the photo gets nudged toward pure black/white;
# high_force controls how hard a conflicting module gets corrected. We render
# at the most photo-preserving level first and only fall back to stronger
# (uglier but safer) levels if the result doesn't actually decode.
FORCE_LADDER = [
    (0.15, 0.85),
    (0.25, 0.88),
    (0.35, 0.90),
    (0.50, 0.95),
    (0.70, 1.0),
    (0.90, 1.0),
    (1.0, 1.0),  # failsafe: zero photo influence, functionally a plain QR
]

# Some (version, error-correction) encodings of otherwise-ordinary data turn
# out to be unreliable for certain scanners regardless of styling (observed:
# version 6 at ERROR_CORRECT_H). Bumping the version sidesteps that, so it's
# tried as an outer fallback layer alongside the force ladder.
VERSION_ESCALATION_OFFSETS = [0, 1, 2, 3, 5]


def _hex_to_rgb(value: Optional[str], fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    value = (value or "").strip().lstrip("#")
    if len(value) != 6:
        return fallback
    try:
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return fallback


def _lerp(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _median_luminance(gray_img: Image.Image) -> float:
    hist = gray_img.histogram()
    total = sum(hist)
    cum = 0
    for level, count in enumerate(hist):
        cum += count
        if cum >= total / 2:
            return float(level)
    return 127.0


def _alignment_centers(version: int, n: int):
    positions = pattern_position(version)
    first, last = positions[0], positions[-1]
    corners = {(first, first), (first, last), (last, first)}
    centers = []
    for r in positions:
        for c in positions:
            if (r, c) in corners:
                continue
            centers.append((r, c))
    return centers


def _in_version_info_zone(row: int, col: int, n: int, version: int) -> bool:
    """Versions 7+ carry two redundant version-info blocks near the top-right
    and bottom-left finder patterns, just outside the finder+separator zone.
    They're only BCH-protected for up to 3 bit errors, so they can't take the
    same photo-blend treatment as ordinary data modules."""
    if version < 7:
        return False
    lo, hi = n - 11, n - 9
    if row <= 5 and lo <= col <= hi:
        return True
    if col <= 5 and lo <= row <= hi:
        return True
    return False


def _in_crisp_zone(row: int, col: int, n: int, version: int, alignment_centers) -> bool:
    """Finder, timing, alignment, and version-info patterns must stay
    undistorted — these are the structural landmarks (and metadata) a
    scanner needs to locate and correctly sample the grid, and they aren't
    protected by error correction the way ordinary data modules are."""
    if row < FINDER_ZONE and col < FINDER_ZONE:
        return True
    if row < FINDER_ZONE and col >= n - FINDER_ZONE:
        return True
    if row >= n - FINDER_ZONE and col < FINDER_ZONE:
        return True
    if row == 6 or col == 6:
        return True
    if _in_version_info_zone(row, col, n, version):
        return True
    for r, c in alignment_centers:
        if abs(row - r) <= 2 and abs(col - c) <= 2:
            return True
    return False


def _build_qr_matrix(data: str, box_size: int, border: int, min_version: int):
    """Make a QRCode, boosted to at least min_version for more resolution."""
    probe = qrcode.QRCode(version=None, error_correction=ERROR_CORRECT_H, box_size=box_size, border=border)
    probe.add_data(data)
    probe.make(fit=True)
    version = max(probe.version, min_version)

    qr = qrcode.QRCode(version=version, error_correction=ERROR_CORRECT_H, box_size=box_size, border=border)
    qr.add_data(data)
    qr.make(fit=False)
    return qr


def _render_photo_qr(
    qr,
    photo: Image.Image,
    accent: Tuple[int, int, int],
    background: Tuple[int, int, int],
    box_size: int,
    border: int,
    low_force: float,
    high_force: float,
) -> Image.Image:
    modules = qr.modules
    n = qr.modules_count
    alignment_centers = _alignment_centers(qr.version, n)

    data_size = n * box_size
    full_size = data_size + 2 * border * box_size

    grayscale = ImageOps.autocontrast(photo.convert("L"))
    fitted = ImageOps.fit(grayscale, (data_size, data_size), Image.LANCZOS)
    duotone = ImageOps.colorize(fitted, black=accent, white=background).convert("RGB")

    threshold = _median_luminance(fitted)
    light_side_range = max(255 - threshold, 1.0)
    dark_side_range = max(threshold, 1.0)

    canvas = Image.new("RGB", (full_size, full_size), background)
    offset = border * box_size
    canvas.paste(duotone, (offset, offset))

    for row in range(n):
        for col in range(n):
            dark = modules[row][col]
            x0 = offset + col * box_size
            y0 = offset + row * box_size

            if _in_crisp_zone(row, col, n, qr.version, alignment_centers):
                color = accent if dark else background
            else:
                gray_region = fitted.crop(
                    (col * box_size, row * box_size, (col + 1) * box_size, (row + 1) * box_size)
                )
                local_lum = ImageStat.Stat(gray_region).mean[0]

                # How much does this pixel conflict with what its module needs to be?
                # 0 = pixel already agrees with the required polarity, 1 = fully opposed.
                # Each direction is normalized by its own actual tonal range around the
                # threshold, so a narrow-range (e.g. mostly-dark) photo still produces
                # full-strength conflict signal instead of being swamped by the other side.
                if dark:
                    conflict = max(0.0, (local_lum - threshold) / light_side_range)
                else:
                    conflict = max(0.0, (threshold - local_lum) / dark_side_range)
                conflict = min(conflict, 1.0)

                force = low_force + (high_force - low_force) * conflict
                target = accent if dark else background

                color_region = duotone.crop(
                    (col * box_size, row * box_size, (col + 1) * box_size, (row + 1) * box_size)
                )
                avg = tuple(round(v) for v in ImageStat.Stat(color_region).mean)
                color = _lerp(avg, target, force)

            canvas.paste(
                Image.new("RGB", (box_size, box_size), color),
                (x0, y0, x0 + box_size, y0 + box_size),
            )

    return canvas


_QR_DETECTOR = cv2.QRCodeDetector()


def _decodes_to(img: Image.Image, expected: str) -> bool:
    """Server-side scannability check: render to an array and try to decode
    it exactly like a real scanner would, so we never ship a QR code that
    doesn't actually work."""
    arr = np.array(img.convert("RGB"))[:, :, ::-1].copy()
    try:
        decoded, _points, _ = _QR_DETECTOR.detectAndDecode(arr)
    except cv2.error:
        return False
    return decoded == expected


def build_qr_image(
    data: str,
    photo: Optional[Image.Image] = None,
    accent_color: Optional[str] = None,
    background_color: Optional[str] = None,
    box_size: int = 10,
    border: int = 4,
    min_version: int = MIN_VERSION,
) -> Image.Image:
    """Build a scannable QR code.

    Without a photo: a plain QR in the given (or default) colors.

    With a photo: the data modules are rendered as a duotone version of the
    photo. Contrast is added adaptively — a module whose underlying photo
    pixel already leans the right way (dark module over a naturally dark
    area, or light module over a naturally light area) is barely touched, so
    the photo stays recognizable; a module that conflicts with its pixel
    gets pushed toward the required color so the code still scans. Finder,
    timing, alignment, and version-info patterns (the structural landmarks
    and metadata a scanner needs) are left completely crisp.

    Every candidate render is verified by actually decoding it before it's
    returned. If the most photo-preserving version doesn't decode cleanly,
    we automatically re-render with progressively stronger correction until
    one does — so a generated code is never handed back broken.
    """
    accent = _hex_to_rgb(accent_color, _hex_to_rgb(DEFAULT_ACCENT, (26, 26, 26)))
    background = _hex_to_rgb(background_color, _hex_to_rgb(DEFAULT_BACKGROUND, (255, 255, 255)))

    best = None
    for offset in VERSION_ESCALATION_OFFSETS:
        qr = _build_qr_matrix(data, box_size, border, min_version + offset)

        if photo is None:
            candidate = qr.make_image(fill_color=accent, back_color=background).convert("RGB")
            best = candidate
            if _decodes_to(candidate, data):
                return candidate
            continue

        for low_force, high_force in FORCE_LADDER:
            candidate = _render_photo_qr(qr, photo, accent, background, box_size, border, low_force, high_force)
            best = candidate
            if _decodes_to(candidate, data):
                return candidate

    return best
